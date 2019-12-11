# A script for fine-tuning the given model.

import cv2
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from data_handler import extractDataset
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
EPOCHS = 1 
BATCH_SIZE = 16
IMG_SIZE = (128, 128)

opt = Adam(lr = INIT_LR, decay = INIT_LR / EPOCHS)
pretrained.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

# load the dataset

print("!! loading dataset...")

# run training

trainFileNames, trainSet = extractDataset(args.dataset, BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1])

print("!! fine-tuning dataset...")
pretrained.fit(trainSet,
    epochs = EPOCHS)

# run testing 

print("!! testing on test set...")
testFiles, testSet = extractDataset(args.test_set, BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1])
print(len(testFiles), "samples loaded")
results = pretrained.evaluate(testSet)
print("evaluator:", results)
# save the model

print("!! saving model...")
pretrained.save(args.to_save)



