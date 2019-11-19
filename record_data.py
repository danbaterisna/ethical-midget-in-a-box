import numpy as np
import time
import os, itertools
import cv2
import argparse as ap
from matplotlib import pyplot as plt

def processFrame(frame, net, minConf, frameCount, savePath):
    (h, w) = frame.shape[:2]
    preprocessed = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))
    net.setInput(preprocessed)
    detection = net.forward()
    for i in range(0, detection.shape[2]):
        conf = detection[0, 0, i, 2]
        if conf > minConf:
            print(conf)
            boundingBox = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
            (sx, sy, ex, ey) = boundingBox.astype('int')
            faceROI = frame[sy:ey, sx:ex]
            imagePath = os.path.join(savePath, f"{frameCount}.png")
            cv2.imwrite(imagePath, faceROI)
            print(imagePath, "saved")
    return frame

def main(args):
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    dim = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dim = (int(dim[0]), int(dim[1]))
    print("loading model...")
    prototxtPath = os.path.join(args.model, "deploy.prototxt.txt")
    modelPath = os.path.join(args.model, "res10_300x300_ssd_iter_140000.caffemodel")
    net = cv2.dnn.readNetFromCaffe(prototxtPath, modelPath)
    if args.save_to:
        writer = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.save_to, writer, fps/3, dim)
    print(f"Reading input at {fps} fps")
    print(f"Frame is {dim}")
    # video processing loop
    timeOfLoop = time.time()
    for i in itertools.count():
        key = cv2.waitKey(1)
        ret, frame = cap.read()
        if i % args.pace == 0:
            frameOut = processFrame(frame, net, args.confidence, i//args.pace, args.save_to)
        else:
            frameOut = frame
        cv2.imshow("stream", frameOut)
        if key & 0xFF == ord('q'):
            break
        timeOfLoop = time.time()

    cap.release()
    if args.save_to:
        out.release()
    cv2.destroyAllWindows()

parser = ap.ArgumentParser(description="Starts a video stream.")

parser.add_argument('-s', '--save_to', type=str, help='Directory to save to.')
parser.add_argument('-m', '--model', type=str, required=True, help='path to model files')
parser.add_argument('-c', '--confidence', type=float, default=0.5, help='minimum required detection confidence')
parser.add_argument('-p', '--pace', type=int, default=10, help='number of delay frames')

args = parser.parse_args()
main(args)

