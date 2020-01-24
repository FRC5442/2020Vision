import cv2 as cv
import numpy as np
import time

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    cv.imshow('frame', hsv)
    if cv.waitKey(1) == ord('q'):
        break

    cap.release()
    cv.destroyAllWindows()