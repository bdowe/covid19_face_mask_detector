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


