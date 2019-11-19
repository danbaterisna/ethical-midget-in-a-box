from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import argparse, pickle, time, os
import cv2

def processFrame(frame, detectNet, liveModel, liveLabels, minConf):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))

    detectNet.setInput(blob)
    detections = detectNet.forward()

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > minConf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (sx, sy, ex, ey) = box.astype("int")
            sx = max(0, sx)
            ex = min(w, ex)
            sy = max(0, sy)
            ey = min(h, ey)

            # get the face ROI
            face = frame[sy:ey, sx:ex]
            if face.shape[0] == 0:
                continue
            face = cv2.resize(face, (64, 64))
            face = face.astype('float') / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # pass it through the liveness detector
            preds = liveModel.predict(face)[0]
            j = np.argmax(preds)
            label = liveLabels.classes_[j]

            boxColor = (0, 255, 0) if label == "real" else (0, 0, 255)
            frame = cv2.rectangle(frame, (sx, sy), (ex, ey), boxColor, int(2 + 6 * preds[j]))

    return frame


def main(args):
    prototxtPath = os.path.join(args.detector, "deploy.prototxt.txt")
    modelPath = os.path.join(args.detector, "res10_300x300_ssd_iter_140000.caffemodel")
    detectNet = cv2.dnn.readNetFromCaffe(prototxtPath, modelPath)

    print("!! loading deep learning hax...")
    liveModel = load_model(args.model)
    labels = None
    with open(args.labels, "rb") as labelFile:
        labels = pickle.loads(labelFile.read())

    print("!! info starting stream...")
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    dim = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dim = (int(dim[0]), int(dim[1]))

    while True:
        key = cv2.waitKey(1)
        ret, frame = cap.read()
        frameOut = processFrame(frame, detectNet, liveModel, labels, args.confidence)
        cv2.imshow("stream", frameOut)
        if key & 0xFF == ord("q"):
            break

parser = argparse.ArgumentParser(description="Magical liveness hackery.")
parser.add_argument("-m", "--model", type=str, required=True, help="path to trained model")
parser.add_argument("-l", "--labels", type=str, required=True, help="path to label encoder")
parser.add_argument("-d", "--detector", type=str, required=True, help="path to face detector")
parser.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum detection prob")

args = parser.parse_args()
main(args)
