# Runs 10-fold cross validation on the dataset.
# Parameters:
# - the dataset
# - the model to refine-tune
# - the number of layers to freeze [the current FT suggests 2]

import cv2
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import KFold
import random
import argparse, os
from data_handler import extractDataset, filenamesToDataset

parser = argparse.ArgumentParser(description = "A script to conduct cross-validation.")
parser.add_argument("dataset", type=str, help="Path to the dataset")
parser.add_argument("k", type=int, help="Number of folds to use")
parser.add_argument("model", type=str, help="Base model to use for fine-tuning.")
parser.add_argument("freeze_count", type=int, help="Fine-tuning parameter.")

args = parser.parse_args()

def prepareHackFolder(fileNameList, hackFolderName):
    os.system(f"mkdir {hackFolderName}/; mkdir {hackFolderName}/real; mkdir {hackFolderName}/fake")
    for fileName in fileNameList:
        #get the last part of the 
        if fileName.split(os.path.sep)[-2] == "real":
            os.system(f"cp {fileName} {hackFolderName}/real/")
        else:
            os.system(f"cp {fileName} {hackFolderName}/fake/")



def conductRound(trainFiles, testFiles):
    # load model
    print("!! loading model...")
    pretrained_raw = load_model(args.model)

    pt_second = pretrained_raw.layers[-2].output
    newOutput = Dense(2, activation="softmax", name="predictions_new")(pt_second)

    pretrained = Model(inputs = pretrained_raw.input, outputs=newOutput)

    # freeze first few layers
    print("!! configuring for fine-tuning...")

    for layer in pretrained.layers[:args.freeze_count]:
        layer.trainable = False

    print("!! compiling model...")

    INIT_LR = 1e-4
    EPOCHS = 50
    BATCH_SIZE = 16
    IMG_SIZE = (128, 128)
    HF_NAME = "HAX_currentRound/"

    print("!! extracting DS")

    prepareHackFolder(trainFiles, HF_NAME)
    trainFiles, trainDS = extractDataset(HF_NAME, BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1])

    print("DEBUG: ")
    print(trainDS)

    opt = Adam(lr = INIT_LR, decay = INIT_LR / EPOCHS)
    pretrained.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    pretrained.fit(trainDS, epochs = EPOCHS)

    os.system(f"rm -r {HF_NAME}")

    print("!! finished training. now evaluating...")

    prepareHackFolder(testFiles, HF_NAME)

    testFiles, testDS = extractDataset(HF_NAME, BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1])
    results = pretrained.evaluate(testDS)

    os.system(f"rm -r {HF_NAME}")

    print("!! results:", results)

print("!!!! loading dataset...")


fileNameList = [f for f in glob.glob(args.dataset + "/real/*.*")] + [f for f in glob.glob(args.dataset + "/fake/*.*")]
fileNameList = np.array(fileNameList)
random.shuffle(fileNameList)

dataIndices = list(range(len(fileNameList)))

kfold = KFold(args.k, True, 42)
currentRound = 1
for trainI, testI in kfold.split(dataIndices):
    print(f"!!!! running cv round {currentRound}/{args.k}")
    conductRound((fileNameList[trainI]), (fileNameList[testI]))
    currentRound += 1


