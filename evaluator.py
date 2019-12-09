import cv2
import os, argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import glob

parser = argparse.ArgumentParser(description = "Given a dataset, output a confusion matrix for that dataset.")

parser.add_argument("model", type=str, help="Path to the trained model file.")
parser.add_argument("dataset", type=str, help="Path to the dataset.")

args = parser.parse_args()

liveModel = load_model(args.model)

def isLive(img):
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float') / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    results = liveModel.predict(img)
    return np.argmax(results[0]) == 0

# get a listing of all image filenames
filesToTest = [f for f in glob.glob(args.dataset + "/real/*.*")] + [f for f in glob.glob(args.dataset + "/fake/*.*")]

# extract fake/real ground truth, get model prediction
trueState = [fileName.split(os.path.sep)[-2] == "real" for fileName in filesToTest]
claimedState = [isLive(cv2.imread(fileName)) for fileName in filesToTest]

# tally 
matrixCells = list(zip(trueState, claimedState))
TP = matrixCells.count((True, 1))
TN = matrixCells.count((False, 0))
FP = matrixCells.count((False, 1))
FN = matrixCells.count((True, 0))

# print table

print("Total samples:", len(filesToTest))
print("Correctly ID-ed as real:", TP, f"{TP/len(filesToTest):.4f}")
print("Correctly ID-ed as fake:", TN, f"{TN/len(filesToTest):.4f}")
print("ID-ed as real, but was actually fake:", FP, f"{FP/len(filesToTest):.4f}")
print("ID-ed as fake, but was actually real:", FN, f"{FN/len(filesToTest):.4f}")

