# import packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np 
import argparse
import imutils
import time
import cv2
import os
from helpers import detect_and_predict_mask

# construct the argument parser and parse args
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize video stream and allow camera sensor to warm up
print(["[INFO] starting video stream..."])
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over frames from video stream
while True:
	# grab the frame from the threaded video stream and resize to have max width of 400px
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a face mask or not
	locs, preds = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over detected face locations and their corresponding predictions
	for box, pred in zip(locs, preds):
		# unpack bb and preds
		startX, startY, endX, endY = box
		mask, withoutMast = pred

		# determine class label and color we'll use to draw bb and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on output frame
		cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key was pressed, break out of loop
	if key == ord("q"):
		break	

# cleanup
cv2.destroyAllWindows()
vs.stop()

