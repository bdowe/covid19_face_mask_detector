from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import os

#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output face mask detector model")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="path to output face mask detector model")
args = vars(ap.parse_args())

# initialize hyper parameters
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# grab list of images, then initialize list of data and class images
print("[INFO] loading_img...")
imagePaths = list(paths.list_images(args['dataset']))
data = []
labels = []

# loop over image paths
for imagePath in imagePaths:
	# extract class label from filename
	label = imagePath.split(os.path.sep)[-2]

	# load input image and preprocess it (224x224)
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# update data and labels lists
	data.appen(image)
	labels.append(label)

# convert data and labels (features and targets) to numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)


