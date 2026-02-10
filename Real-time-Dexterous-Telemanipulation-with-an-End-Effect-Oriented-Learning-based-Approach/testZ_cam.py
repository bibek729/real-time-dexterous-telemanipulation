import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import transformations as tf

import gymnasium as gym
import gymnasium_robotics

from rl_modules.models import actor
from arguments import get_args


def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std + 1e-8), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std + 1e-8), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    return torch.tensor(inputs, dtype=torch.float32)


def unwrap_angle(curr, prev):
    """Keep curr close to prev by adding/subtracting 2*pi."""
    if prev is None:
        return curr
    while curr - prev > np.pi:
        curr -= 2 * np.pi
    while curr - prev < -np.pi:
        curr += 2 * np.pi
    return curr


if __name__ == "__main__":
    args = get_args()

    # -----------------------------
    # 1) Load camera calibration
    # -----------------------------
    if not os.path.exists("camera_params.npz"):
        raise FileNotFoundError("camera_params.npz not found in current folder.")

    loaded = np.load("camera_params.npz")
    camera_matrix = loaded["mtx"].astype(np.float32)
    dist_coeffs = loaded["dist"].astype(np.float32)

    print("Loaded camera_params.npz ✅")

    # -----------------------------
    # 2) ArUco setup (OpenCV 4.7+)
    # -----------------------------
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    MARKER_LENGTH = 0.05  # meters (match your printed marker size!)
    AXIS_LENGTH = 0.03

    # -----------------------------
    # 3) Open camera
    # -----------------------------
    cap = cv2.VideoCapture(1)  # try 0 if 1 doesn't work
    if not cap.isOpened():
        raise RuntimeError("Camera did not open. Try changing VideoCapture(1) -> VideoCapture(0).")

    # -----------------------------
    # 4) Load model
    # -----------------------------
    model_path = os.path.join(args.save_dir, args.env_name, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # PyTorch 2.6+ safety change: we explicitly allow full load
    o_mean, o_std, g_mean, g_std, model_state = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )
    print(f"Loaded model ✅: {model_path}")

    # -----------------------------
    # 5) Create env
    # -----------------------------
    env = gym.make(args.env_name, render_mode="human")

    obs_dict, info = env.reset()
    env_params = {
        "obs": obs_dict["observation"].shape[0],
        "goal": obs_dict["desired_goal"].shape[0],
        "action": env.action_space.shape[0],
        "action_max": env.action_space.high[0],
    }

    actor_net = actor(env_params)
    actor_net.load_state_dict(model_state)
    actor_net.eval()

    # -----------------------------
    # Tracking buffers
    # -----------------------------
    target_angles = []
    actual_angles = []

    prev_target = None
    prev_actual = None
    roll_filtered = None
    alpha = 0.15  # smoothing factor (0.05 smoother, 0.3 faster)

    print("Running... press 'q' in the camera window to quit.")

    # -----------------------------
    # Main loop
    # -----------------------------
    for episode in range(1):  # keep 1 for debugging; increase later
        obs_dict, info = env.reset()
        obs = obs_dict["observation"]

        # NOTE: DO NOT set env.unwrapped.initial_qpos unless you have correct length (v1 uses 38)
        # env.unwrapped.initial_qpos = ...

        for t in range(300):
            # ----- Camera frame -----
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = detector.detectMarkers(gray)

            roll_rad = prev_target if prev_target is not None else 0.0  # default if not detected

            if ids is not None and len(ids) > 0:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, MARKER_LENGTH, camera_matrix, dist_coeffs
                )

                # Use the first detected marker
                rvec = rvecs[0]
                tvec = tvecs[0]

                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, AXIS_LENGTH)
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                # rvec -> rotation matrix
                R, _ = cv2.Rodrigues(rvec)

                # Euler angles (degrees) from rotation matrix
                # cv2.RQDecomp3x3 returns (anglesX, anglesY, anglesZ) in degrees
                angles_deg = cv2.RQDecomp3x3(R)[0]
                roll_deg = -float(angles_deg[2])  # match your previous sign convention
                roll_rad = np.deg2rad(roll_deg)

                # unwrap to avoid +/-pi flips
                roll_rad = unwrap_angle(roll_rad, prev_target)

                # smooth (low-pass filter)
                if roll_filtered is None:
                    roll_filtered = roll_rad
                else:
                    roll_filtered = (1 - alpha) * roll_filtered + alpha * roll_rad

                roll_rad = roll_filtered

                prev_target = roll_rad

            # Show camera
            cv2.imshow("ARUCO Pose Estimation", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise SystemExit

            # ----- Build goal g -----
            quaternion = tf.quaternion_from_euler(0.0, 0.0, roll_rad)

            # IMPORTANT:
            # Your env goal is (pos + quat). For this test, we keep pos constant.
            pos = np.array([1.01570427, 0.87487394, 0.17090474], dtype=np.float32)
            g = np.concatenate([pos, quaternion]).astype(np.float32)

            # For v1, set goal directly
            if hasattr(env.unwrapped, "goal"):
                env.unwrapped.goal = g.copy()
            elif hasattr(env.unwrapped, "_goal"):
                env.unwrapped._goal = g.copy()

            # ----- Policy action -----
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_net(inputs)
            action = pi.cpu().numpy().squeeze()

            # ----- Step env -----
            obs_new, reward, terminated, truncated, info = env.step(action)
            obs = obs_new["observation"]

            # achieved_goal quaternion is last 4 values
            acg = obs_new["achieved_goal"]
            _, _, oz = tf.euler_from_quaternion(acg[-4:])

            # unwrap actual too (so plot is stable)
            oz = unwrap_angle(oz, prev_actual)
            prev_actual = oz

            target_angles.append(roll_rad)
            actual_angles.append(oz)

        # ----- Plot -----
        plt.figure()
        plt.plot(target_angles, label="Target")
        plt.plot(actual_angles, label="Actual")
        plt.title("Target tracking test")
        plt.xlabel("Timestep")
        plt.ylabel("Angle (rad)")
        plt.legend()
        plt.show()

    cap.release()
    cv2.destroyAllWindows()
    env.close()
