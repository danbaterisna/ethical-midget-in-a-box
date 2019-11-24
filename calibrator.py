# A script for fine-tuning the given model.

import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import argparse, os

# parse the arguments

parser = argparse.ArgumentParser(description = "A script for fine-tuning the given model.")
parser.add_argument("dataset", type=str, help="A path to the ds to use for fine-tuning.")
parser.add_argument("model", type=str, help="File name of pre-trained model.")
parser.add_argument("to_save", type=str, help="File name to save the model to.")
parser.add_argument("freeze_count", type=int, help="Number of bottom layers to freeze.")

args = parser.parse_args()

# load the model
print("!! loading model...")
pretrained_raw = load_model(args.model)

# freeze first few layers
print("!! configuring for fine-tuning...")

pt_second = pretrained_raw.layers[-2].output
newOutput = Dense(2, activation="sigmoid", name="predictions_new")(pt_second)

pretrained = Model(inputs=pretrained_raw.input, outputs=newOutput)

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
filename_ds = tf.data.Dataset.list_files(os.path.join(args.dataset, "*/*.jpg"))
filename_train_ds = filename_ds.take(3000)
filename_val_ds = filename_ds.skip(3000)

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
    ds = ds.batch(BATCH_SIZE)
    return ds

labeled_train_ds = filename_train_ds.map(process_path)
labeled_val_ds = filename_val_ds.map(process_path)

labeled_train_ds = prepare(labeled_train_ds)
labeled_val_ds = prepare(labeled_val_ds)

# run training

print("!! fine-tuning dataset...")
pretrained.fit(labeled_train_ds,
    epochs = EPOCHS,
    validation_data = labeled_val_ds)

# save the model

print("!! saving model...")
pretrained.save(args.to_save)



