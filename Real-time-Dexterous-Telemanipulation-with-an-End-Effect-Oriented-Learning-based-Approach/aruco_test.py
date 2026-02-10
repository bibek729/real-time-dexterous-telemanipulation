import cv2
import numpy as np

# If you have real calibration, you can use it.
# For now we use a simple pinhole model like your code.
camera_matrix = np.array([[640, 0, 320],
                          [0, 640, 240],
                          [0,   0,   1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

cap = cv2.VideoCapture(1)   # if black screen, change 1 -> 0

MARKER_LENGTH = 0.05  # meters (5 cm). Change to your printed marker size.

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)



    if ids is not None and len(ids) > 0:
        # Pose estimation in new API
        rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH, camera_matrix, dist_coeffs
        )

        for rvec, tvec in zip(rvecs, tvecs):
            # Draw XYZ axes (new OpenCV function)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow("ARUCO Pose Estimation", frame)

    # ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
