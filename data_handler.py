import tensorflow as tf
import os, glob

def filenamesToDataset(fileNameList, batch_size, imgx, imgy):
    fileNameDS = tf.data.Dataset.from_generator(lambda: fileNameList, tf.string)
    print(fileNameDS)

    def extract_img(img):
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [imgx, imgy])
        return img

    def get_label(fileName):
        parts = tf.strings.split(fileName, os.path.sep)
        return [parts[-2] == "real", parts[-2] == "fake"]

    def process_path(fileName):
        img = tf.io.read_file(fileName)
        img = extract_img(img)
        label = get_label(fileName)
        return img, label

    def prepare(ds):
        ds = ds.shuffle(buffer_size = 1000)
        ds = ds.batch(batch_size)
        return ds

    imageData = fileNameDS.map(process_path)
    imageData = prepare(imageData)

    return fileNameList, imageData

def extractDataset(path, batch_size, imgx, imgy):
    fileNameDS = tf.data.Dataset.list_files(os.path.join(path, "*/*.jpg"))
    fileNameList = [f for f in glob.glob(path + "/real/*.*")] + [f for f in glob.glob(path + "/fake/*.*")]

    def extract_img(img):
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [imgx, imgy])
        return img

    def get_label(fileName):
        parts = tf.strings.split(fileName, os.path.sep)
        return [parts[-2] == "real", parts[-2] == "fake"]

    def process_path(fileName):
        img = tf.io.read_file(fileName)
        img = extract_img(img)
        label = get_label(fileName)
        return img, label

    def prepare(ds):
        ds = ds.shuffle(buffer_size = 1000)
        ds = ds.batch(batch_size)
        return ds

    imageData = fileNameDS.map(process_path)
    imageData = prepare(imageData)

    return fileNameList, imageData

