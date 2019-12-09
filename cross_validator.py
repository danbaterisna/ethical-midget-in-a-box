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

parser = argparse.ArgumentParser(description = "A script to conduct cross-validation.")
parser.add_argument("dataset", type=str, help="Path to the dataset")
parser.add_argument("k", type=int, help="Number of folds to use")
parser.add_argument("model", type=str, help="Base model to use for fine-tuning.")
parser.add_argument("freeze_count", type=int, help="Fine-tuning parameter.")

args = parser.parse_args()

def conductRound(trainSet, testSet):
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

    opt = Adam(lr = INIT_LR, decay = INIT_LR / EPOCHS)
    pretrained.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    pretrained.fit(trainSet, batch_size = BATCH_SIZE, epochs = EPOCHS)

    print("!! finished training. now evaluating...")
    results = pretrained.evaluate(testSet, batch_size = BATCH_SIZE)

    print("!! results:", results)

print("!!!! loading dataset...")


fileNameList = [f for f in glob.glob(args.dataset + "/real/*.*")] + [f for f in glob.glob(args.dataset + "/fake/*.*")]
random.shuffle(fileNameList)

def fetchImage(fileName):
    img = cv2.imread(fileName)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

trueState = [[fileName.split(os.path.sep)[-2] == "real", fileName.split(os.path.sep)[-2] == "fake"] for fileName in fileNameList]
imgData = [fetchImage(fileName) for fileName in fileNameList]
print("!!!! ds loaded, now rescaling...")

#print("!! DEBUGGING: checking labels:")

#for fileName, img, label in zip(fileNameList, imgData, trueState):
#    cv2.imshow(fileName + str(label), img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

imgData = np.array(imgData, dtype = 'float') / 255.0
trueState = np.array(trueState, dtype = 'float')

finalData = np.array(list(zip(imgData, trueState)))

kfold = KFold(args.k, True, 42)
currentRound = 1
for train, test in kfold.split(finalData):
    print(f"!!!! running cv round {currentRound}/{args.k}")
    conductRound(train, test)
    currentRound += 1


