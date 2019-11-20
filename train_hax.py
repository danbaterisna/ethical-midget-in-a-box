# train the hax
import matplotlib
matplotlib.use("Agg")

import tensorflow as tf
from deep_learning_hax import DeepLearningHax
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from keras.utils import np_utils
from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import argparse, pickle, os
import cv2

# construct the hax
def main(args):
    INIT_LR = 5e-4
    BS = 16
    EPOCHS = 50
    IMG_SIZE = (128, 128)

#    imagePaths = list(paths.list_images(args.dataset))
#    data = []
#    labels = []

    # print("!! loading data...")
    # for imagePath in imagePaths:
    #    label = imagePath.split(os.path.sep)[-2]
    #    # test other color spaces: CMYK? grayscale?
    #    image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    #    if image is None or image[0][0][0] == 255:
    #        continue
    #    # retrain with this size parameter
    #    image = cv2.resize(image, IMG_SIZE)
#
#        data.append(image)
#        labels.append(label)
#        print(imagePath, "labeled", label)
#
#    data = np.array(data, dtype="float") / 255.0
#
#    le = LabelEncoder()
#    labels = le.fit_transform(labels)
#    labels = np_utils.to_categorical(labels, 2)
#
#    # try diff values of random_state
#    print("!! setting up train-test split")
#    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state=42)
#
#    # write filenames of trainX, testX to test files
#    print("!! preparing augmenter [SKIPPED]")

    # adjust numbers appropriately
    print("!! loading filenames...")
    filename_ds = tf.data.Dataset.list_files(os.path.join(args.dataset, "*/*.jpg"))
    filename_train_ds = filename_ds.take(6000)
    filename_val_ds = filename_ds.skip(6000).take(1000)
    filename_test_ds = filename_ds.skip(7000)

    def extract_img(img):
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [IMG_SIZE[0], IMG_SIZE[1]])
        return img

    def get_label(fileName):
        parts = tf.strings.split(fileName, os.path.sep, result_type="RaggedTensor")
        return parts[-2] == "real"

    def process_path(fileName):
        img = tf.io.read_file(fileName)
        img = extract_img(img)
        label = get_label(fileName)
        return img, label

    def prepare(ds):
        ds = ds.shuffle(buffer_size = 1000)
        ds = ds.batch(BS)
        return ds

    print("!! loading dataset")
    labeled_test_ds = filename_test_ds.map(process_path)
    labeled_train_ds = filename_train_ds.map(process_path)
    labeled_val_ds = filename_val_ds.map(process_path)

    labeled_test_ds = prepare(labeled_test_ds)
    labeled_train_ds = prepare(labeled_train_ds)
    labeled_val_ds = prepare(labeled_val_ds)

    print("!! compiling network...")
    opt = Adam(lr=INIT_LR, decay = INIT_LR/EPOCHS)
    model = DeepLearningHax.build(width=IMG_SIZE[0], height=IMG_SIZE[1], depth=3, classes=2)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    print(f"!! training network for {EPOCHS} epochs...")
    # H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
    #        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
    #        epochs=EPOCHS)
    H = model.fit(labeled_train_ds,
           validation_data=labeled_val_ds,
           epochs=EPOCHS)

    # results of this are too high; check data hygiene
    print("!! evaluating network [TODO]...")
    #predictions = model.predict(labeled_test_ds, batch_size=BS)
    #print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names = le.classes_))

    print(f"!! saving network...")
    model.save(args.model)

    with open(args.labels, "wb") as f:
        f.write(pickle.dumps(le))


# argparse hax
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="path to dataset")
parser.add_argument("-m", "--model", required=True, type=str, help="path to model file")
parser.add_argument("-l", "--labels", required=True, type=str, help="path to label encoder")
parser.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")

args = parser.parse_args()
main(args)
