#!/usr/bin/env python3
#----------------------------------------------------------------------------
# Copyright (c) 2018 FIRST. All Rights Reserved.
# Open Source Software - may be modified and shared by FRC teams. The code
# must be accompanied by the FIRST BSD license file in the root directory of
# the project.
#
# Screaming Chickens 2019 license: use it as much as you want. Crediting is recommended because it lets me know that I am being useful.
# Credit to Screaming Chickens 3997
#----------------------------------------------------------------------------

import json
import time
import sys
from threading import Thread

from cscore import CameraServer, MjpegServer, VideoSource, UsbCamera
from networktables import NetworkTablesInstance
from networktables import NetworkTables
import ntcore
import cv2
import numpy as np
import math
################################ SCREAMING CHICKENS CODE ############################

import datetime

class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0

	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self

	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()

	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1

	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()

	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()


# class that runs separate thread for showing video,
class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, imgWidth, imgHeight, cameraServer, frame=None, name='stream'):
        self.outputStream = cameraServer.putVideo(name, imgWidth, imgHeight)
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            self.outputStream.putFrame(self.frame)

    def stop(self):
        self.stopped = True

    def notifyError(self, error):
        self.outputStream.notifyError(error)

class WebcamVideoStream:
    def __init__(self, camera, cameraServer, frameWidth, frameHeight, name="WebcamVideoStream"):
        # initialize the video camera stream and read the first frame
        # from the stream

        #Automatically sets exposure to 0 to track tape
        self.webcam = camera
        self.webcam.setExposureManual(0)
        #Some booleans so that we don't keep setting exposure over and over to the same value
        self.autoExpose = False
        self.prevValue = self.autoExpose
        #Make a blank image to write on
        self.img = np.zeros(shape=(frameWidth, frameHeight, 3), dtype=np.uint8)
        #Gets the video
        self.stream = cameraServer.getVideo(camera = camera)
        (self.timestamp, self.img) = self.stream.grabFrame(self.img)

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            #Boolean logic we don't keep setting exposure over and over to the same value
            if self.autoExpose:

                self.webcam.setExposureAuto()
            else:

                self.webcam.setExposureManual(0)
            #gets the image and timestamp from cameraserver
            (self.timestamp, self.img) = self.stream.grabFrame(self.img)

    def read(self):
        # return the frame most recently read
        return self.timestamp, self.img

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
    def getError(self):
        return self.stream.getError()

################### END OF SCREAMING CHICKENS CODE ########################

################### OPENCV IMAGE PROCESSING ######################

image_width = 256
image_height = 144

# Color Threshold the image
upperGreen = np.array([103, 192, 116])
lowerGreen = np.array([53, 0, 17])

# Blur must be an odd number
greenBlur = 3

coverageArea = 112.5/671.5

diagonalView = math.radians(68.5)

horizontalAspect = 16
verticalAspect = 9

diagonalAspect = math.hypot(horizontalAspect, verticalAspect)

horizontalView = math.atan(math.tan(diagonalView/2) * (horizontalAspect / diagonalAspect)) * 2
verticalView = math.atan(math.tan(diagonalView/2) * (verticalAspect / diagonalAspect)) * 2

H_FOCAL_LENGTH = image_width / (2*math.tan((horizontalView/2)))
V_FOCAL_LENGTH = image_height / (2*math.tan((verticalView/2)))

def flipImage(frame):
    return cv2.flip(frame, -1)

def blur(blurRadius, frame):
    img = frame.copy()
    ksize = int(6 * round(blurRadius) + 1)
    blur = cv2.GaussianBlur(img, (ksize, ksize), round(blurRadius))
    return blur

def threshold_video(lower_color, upper_color, frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_color, upper_color)

    return mask

def findTargets(frame, mask):
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    screenHeight, screenWidth, _ = frame.shape

    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5

    image = frame.copy()

    if len(contours) != 0:
        image = findTape(contours, image, centerX, centerY)
        networkTable.putBoolean("tapeDetected", True)
    else:
        networkTable.putBoolean("tapeDetected", False)

    return image

# Draws Contours and finds center and yaw of vision targets
# centerX is center x coordinate of image
# centerY is center y coordinate of image
def findTape(contours, image, centerX, centerY):
    screenHeight, screenWidth, channels = image.shape
    #Seen vision targets (correct angle, adjacent to each other)
    targets = []

    if len(contours) >= 1:
        #Sort contours by area size (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        biggestCnts = []
        for cnt in cntsSorted:
            # Get moments of contour; mainly for centroid
            M = cv2.moments(cnt)
            # Get convex hull (bounding polygon on contour)
            hull = cv2.convexHull(cnt)
            # Calculate Contour area
            cntArea = cv2.contourArea(cnt)
            # Calculate Contour Perimeter
            cntPerim = cv2.arcLength(cnt, True)
            # calculate area of convex hull
            hullArea = cv2.contourArea(hull)

            rx, ry, rw, rh = cv2.boundingRect(cnt)

            targetArea = cntArea/(rw*rh)
            similarity = (targetArea/coverageArea)*100
            
            # Filters contours based off of size
            if similarity > 50 and similarity < 150: # Checks contour size

                ### MOSTLY DRAWING CODE, BUT CALCULATES IMPORTANT INFO ###
                # Gets the centeroids of contour
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                if(len(biggestCnts) < 13):

                    ##### DRAWS CONTOUR######
                    # Gets rotated bounding rectangle of contour
                    rect = cv2.minAreaRect(cnt)
                    # Creates box around that rectangle
                    box = cv2.boxPoints(rect)
                    # Not exactly sure
                    box = np.int0(box)
                    # Draws rotated rectangle
                    cv2.drawContours(image, [box], 0, (23, 184, 80), 3)


                    # Calculates yaw of contour (horizontal position in degrees)
                    yaw = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
                    networkTable.putNumber("Yaw", yaw)
                    # Calculates pitch of contour (vertical position in degrees)
                    pitch = calculatePitch(cy, centerY, V_FOCAL_LENGTH)
                    networkTable.putNumber("Pitch", pitch)

                    # Draws a vertical white line passing through center of contour
                    cv2.line(image, (cx, screenHeight), (cx, 0), (255, 255, 255))
                    # Draws a white circle at center of contour
                    cv2.circle(image, (cx, cy), 6, (255, 255, 255))

                    # Draws the contours
                    cv2.drawContours(image, [cnt], 0, (23, 184, 80), 1)

                    # Gets the (x, y) and radius of the enclosing circle of contour
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    # Rounds center of enclosing circle
                    center = (int(x), int(y))
                    # Rounds radius of enclosning circle
                    radius = int(radius)
                    # Makes bounding rectangle of contour
                    boundingRect = cv2.boundingRect(cnt)
                    # Draws countour of bounding rectangle and enclosing circle in green
                    cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (23, 184, 80), 1)

                    cv2.circle(image, center, radius, (23, 184, 80), 1)

                    # Appends important info to array
                    if not biggestCnts:
                         biggestCnts.append([cx, cy, cntArea])
                    elif [cx, cy] not in biggestCnts:
                         biggestCnts.append([cx, cy, cntArea])


        # Sorts array based on area (leftmost to rightmost) to make sure contours are adjacent
        biggestCnts = sorted(biggestCnts, key=lambda x: x[2], reverse=True)

        if len(biggestCnts) > 0:
            #x coords of contours
            cx1 = biggestCnts[0][0]

            cy1 = biggestCnts[0][1]
                
            '''
            5442 EDIT START 
            '''
            networkTable.putNumber("OffsetX", (cx1 - centerX))
            networkTable.putNumber("OffsetY", (cy1 - centerY))
            '''
            5442 EDIT END  
            '''

    #Check if there are targets seen
    if (len(targets) > 0):
        networkTable.putBoolean("tapeDetected", True)
        targets.sort(key=lambda x: math.fabs(x[0]))

    else:
        networkTable.putBoolean("tapeDetected", False)

    cv2.line(image, (round(centerX), screenHeight), (round(centerX), 0), (255, 255, 255), 2)

    return image

# Uses trig and focal length of camera to find yaw.
# Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
def calculateYaw(pixelX, centerX, hFocalLength):
    yaw = math.degrees(math.atan((pixelX - centerX) / hFocalLength))
    return round(yaw)


# Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
def calculatePitch(pixelY, centerY, vFocalLength):
    pitch = math.degrees(math.atan((pixelY - centerY) / vFocalLength))
    # Just stopped working have to do this:
    pitch *= -1
    return round(pitch)

################### FRC VISION IMAGE CODE (WEB INTERFACE) #######################
configFile = "/boot/frc.json"

class CameraConfig: pass

team = None
server = False
cameraConfigs = []
cameras = []

def parseError(str):
    """Report parse error."""
    print("config error in '" + configFile + "': " + str, file=sys.stderr)

def readCameraConfig(config):
    """Read single camera configuration."""
    cam = CameraConfig()

    # name
    try:
        cam.name = config["name"]
    except KeyError:
        parseError("could not read camera name")
        return False

    # path
    try:
        cam.path = config["path"]
    except KeyError:
        parseError("camera '{}': could not read path".format(cam.name))
        return False

    # stream properties
    cam.streamConfig = config.get("stream")

    cam.config = config

    cameraConfigs.append(cam)
    return True

def readConfig():
    """Read configuration file."""
    global team
    global server

    # parse file
    try:
        with open(configFile, "rt", encoding="utf-8") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '{}': {}".format(configFile, err), file=sys.stderr)
        return False

    # top level must be an object
    if not isinstance(j, dict):
        parseError("must be JSON object")
        return False

    # team number  
    try:
        team = j["team"]
    except KeyError:
        parseError("could not read team number")
        return False

    # ntmode (optional)
    if "ntmode" in j:
        str = j["ntmode"]
        if str.lower() == "client":
            server = False
        elif str.lower() == "server":
            server = True
        else:
            parseError("could not understand ntmode value '{}'".format(str))

    # cameras
    try:
        cameras = j["cameras"]
    except KeyError:
        parseError("could not read cameras")
        return False
    for camera in cameras:
        if not readCameraConfig(camera):
            return False

    return True

def startCamera(config):
    """Start running the camera."""
    print("Starting camera '{}' on {}".format(config.name, config.path))
    cs = CameraServer.getInstance()
    camera = UsbCamera(config.name, config.path)
    server = cs.startAutomaticCapture(camera=camera, return_server=True)

    camera.setConfigJson(json.dumps(config.config))

    return cs, camera

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]

    # read configuration
    if not readConfig():
        sys.exit(1)

    # start NetworkTables
    ntinst = NetworkTablesInstance.getDefault()
    #Name of network table -- how the raspbery pi communicates with the robot
    networkTable = NetworkTables.getTable("5442Vision")

    if server:
        print("Setting up NetworkTables server")
        ntinst.startServer()
    else:
        print("Setting up NetworkTables client for team {}".format(team))
        ntinst.startClientTeam(team)

    # start cameras
    streams = []
    for config in cameraConfigs:
        cs, cameraCapture = startCamera(config)
        streams.append(cs)
        cameras.append(cameraCapture)

############################### END OF FRC VISION IMAGE CODE #######################
    webcam = cameras[0]
    cameraServer = streams[0]

    # Starts the webcam stream
    cap = WebcamVideoStream(webcam, cameraServer, image_width, image_height).start()

    # Allocates space for the processed image
    img = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8)

    # Shows stream to shuffleboard
    streamViewer = VideoShow(image_width, image_height, cameraServer, frame=img, name="5442Vision").start()

    tape = False
    fps = FPS().start()

    while True:
        timestamp, frame = cap.read()
        
        if timestamp == 0:
            streamViewer.notifyError(cap.getError());

            continue

        if(networkTable.getBoolean("Driver", False)):
            cap.autoExpose = True
            processed = frame
        else:
            cap.autoExpose = False
            img = flipImage(frame)
            #imgBlur = blur(greenBlur, img)
            hsv = threshold_video(lowerGreen, upperGreen, img)
            processed = findTargets(img, hsv)

        networkTable.putNumber("VideoTimestamp", timestamp)
        streamViewer.frame = processed

        fps.update()

        ntinst.flush()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elsapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.stop()))