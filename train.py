# https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
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
	data.append(image)
	labels.append(label)

# convert data and labels (features and targets) to numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# binarize labels (only 2 classes)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# split data into train and tests sets (80/20)
trainX, testX, trainY, testY = train_test_split(
	data, 
	labels,
	test_size=0.20,
	stratify=labels,
	random_state=42
)

# instantiate train image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range = 0.15,
	horizontal_flip = True,
	fill_mode="nearest"
)

# laod MobileNetV2 net, leaving off head fc layer sets
baseModel = MobileNetV2(
	weights="imagenet",
	include_top=False,
	input_tensor=Input(shape=(224, 224, 3))
)

# construct model head that will be appended to the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(1, activation="sigmoid")(headModel)

# appead head FC model to base model
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over layers in base model and freeze them so they won't be updated during training
for layer in baseModel.layers:
	layer.trainable = False

# compile model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(
	loss="binary_crossentropy", 
	optimizer=opt,
	metrics=["accuracy"]
)

# train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS
)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIxs = model.predict(testX, batch_size=BS)

# get prediction index with highest probability as prediction
predIxs = np.argmax(predIxs, axis=1)

# show classification report
print(classification_report(testY.argmax(axis=1), predIxs, target_names=lb.classes_))

# serialize model to disk
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

# plot training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])





