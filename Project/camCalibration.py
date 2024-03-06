import numpy as np
import cv2 as cv
from cv2 import aruco as aruco
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('db\*.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,5), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,5), corners2, ret)
        cv.imshow('img', img)
        # cv.imwrite('results\grid4.png', img)
        cv.waitKey(500)
        # print(objpoints[0].shape)
cv.destroyAllWindows()

#Calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

'''
#Undistortion
img = cv.imread('db\img2.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


#1. Using cv.undistort()

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('results\calibresult4.png', dst)


#2. Using remapping
# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('results\calibresult4.png', dst)

#Re-projection Error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )
'''
# Define the ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)  # You can choose a different dictionary if needed
parameters = aruco.DetectorParameters()

# Initialize the camera
camera = cv.VideoCapture(0)  # Use the appropriate camera index if you have multiple cameras
camera.set(cv.CAP_PROP_FRAME_WIDTH, 1280)  # Set the camera resolution
camera.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# Axis length for drawing
axis_length = 0.1
marker_size = 0.05

while True:
    ret, frame = camera.read()  # Capture a frame from the camera
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    
    # Detect markers
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if ids is not None:
        # Estimate marker poses
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, mtx, dist)
        
        for i in range(len(ids)):
            # Draw marker outline
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Get the rotation and translation vectors
            rvec, tvec = rvecs[i], tvecs[i]
            
            # Project 3D points to image plane
            points, _ = cv.projectPoints(axis_length * np.float32([[0, 0, 0],
                                                                    [1, 0, 0],
                                                                    [0, 1, 0],
                                                                    [0, 0, 1]]),
                                          rvec, tvec, mtx, dist)
            
            points = np.array(points, dtype=np.int32)  # Convert points to numpy array
            
            # Draw coordinate axes
            cv.line(frame, tuple(points[0].ravel()), tuple(points[1].ravel()), (0, 0, 255), 2)  # X-axis (red)
            cv.line(frame, tuple(points[0].ravel()), tuple(points[2].ravel()), (0, 255, 0), 2)  # Y-axis (green)
            cv.line(frame, tuple(points[0].ravel()), tuple(points[3].ravel()), (255, 0, 0), 2)  # Z-axis (blue)
    
    cv.imshow('ArUco Marker Detection', frame)  # Display the frame
    
    if cv.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
        break

camera.release()  # Release the camera
cv.destroyAllWindows()  # Close all OpenCV windows