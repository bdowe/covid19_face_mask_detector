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

def detect_and_predict_mask(frame, faceNet, maskNet):
	# get dimensions of frame, use them to construct blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300))

	# pass blob through face detection network
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# lists of faces, their corresponding locations, and predictions from the face mask net
	faces = []
	locs = []
	preds = []

	# loop over detections
	for i in range(detections.shape[2]):
		# extract confidence associated with detection
		condfidence = detections[0, 0, i, 2]

		# filter out weak detections (below minimum confidence threshold)
		if confidence > args["confidence"]:
			# compute bounding box coordinates
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			startX, startY, endX, endY = box.astype("int")

			# ensure bounding boxes fall within frame dimensions
			startX, startY = max(0, startX), max(0, startY)
			endX, endY = min(w-1, endX), min(h-1, endY)

			
