# A script for fine-tuning the given model.

import cv2
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
import random
import argparse, os

# parse the arguments

parser = argparse.ArgumentParser(description = "A script for fine-tuning the given model.")
parser.add_argument("dataset", type=str, help="A path to the ds to use for fine-tuning.")
parser.add_argument("test_set", type=str, help="A path to the ds to use for testing.")
parser.add_argument("model", type=str, help="File name of pre-trained model.")
parser.add_argument("to_save", type=str, help="File name to save the model to.")
parser.add_argument("freeze_count", type=int, help="Number of bottom layers to freeze.")

args = parser.parse_args()

# load the model
print("!! loading model...")
pretrained_raw = load_model(args.model)

pt_second = pretrained_raw.layers[-2].output
newOutput = Dense(2, activation="softmax", name="predictions_new")(pt_second)

pretrained = Model(inputs = pretrained_raw.input, outputs=newOutput)

# freeze first few layers
print("!! configuring for fine-tuning...")

for layer in pretrained.layers[:args.freeze_count]:
    layer.trainable = False

# compile the model

print("!! compiling model...")
INIT_LR = 1e-4
EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = (128, 128)

opt = Adam(lr = INIT_LR, decay = INIT_LR / EPOCHS)
pretrained.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

# load the dataset

print("!! loading dataset...")

def extractDataset(path):

    fileNameList = [f for f in glob.glob(path + "/real/*.*")] + [f for f in glob.glob(path + "/fake/*.*")]
    random.shuffle(fileNameList)

    def fetchImage(fileName):
        img = cv2.imread(fileName)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    trueState = [[fileName.split(os.path.sep)[-2] == "real", fileName.split(os.path.sep)[-2] == "fake"] for fileName in fileNameList]
    imgData = [fetchImage(fileName) for fileName in fileNameList]
    print("!! ds loaded, now rescaling...")

    #print("!! DEBUGGING: checking labels:")

    #for fileName, img, label in zip(fileNameList, imgData, trueState):
    #    cv2.imshow(fileName + str(label), img)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()

    imgData = np.array(imgData, dtype = 'float') / 255.0
    trueState = np.array(trueState, dtype = 'float')

    print(trueState)

    return (fileNameList, imgData, trueState)

# run training

trainFileNames, trainImages, trainTruth = extractDataset(args.dataset)

print("!! fine-tuning dataset...")
pretrained.fit(trainImages, trainTruth, batch_size = BATCH_SIZE,
    epochs = EPOCHS)

# run testing 

print("!! testing on test set...")
testFiles, testImages, testTruth = extractDataset(args.test_set)
print(len(testTruth), "samples loaded")
results = pretrained.evaluate(testImages, testTruth, batch_size = BATCH_SIZE)
print("evaluator:", results)
# save the model

print("!! saving model...")
pretrained.save(args.to_save)



