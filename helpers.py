import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

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

			# extract face ROI, convert from BGR to RGB channel ordering, resize to 224x224, preprocess
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# add face and bbs to their respective lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make mask prediction if at least one face was detected
	if len(faces) > 0:
		# for faster inference, bake batch predictions on *all* faces at the same time rather than one by one predictions
		preds = maskNet.predict(faces)

	# return 2-tuple of face locations and corresponding predictions
	return locs, preds