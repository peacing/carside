import numpy as np
import tensorflow as tf
from PIL import Image
from glob import glob

IMAGE_DIR = '/Users/paulsingman/projects/fdl/Fundamentals-of-Deep-Learning-Book/images/*jpg'
tfrecords_filename = 'vroom.tfrecords'

filenames = glob(IMAGE_DIR)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def process_img_path(img_path):
    img_part = img_path.split('/')[-1]
    image_id, label, _ = img_part.split('.')
    return int(image_id), int(label)

writer = tf.python_io.TFRecordWriter(tfrecords_filename)


for img_path in filenames:
    
    img = np.array(Image.open(img_path))
    
    image_id, label = process_img_path(img_path)
    
    # The reason to store image sizes was demonstrated
    # in the previous example -- we have to know sizes
    # of images to later read raw serialized string,
    # convert to 1d array and convert to respective
    # shape that image used to have.
    height = img.shape[0]
    width = img.shape[1]
    
    # Put in the original images into array
    # Just for future check for correctness
    
    img_raw = img.tostring()
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_id': _int64_feature(image_id),
        'label': _int64_feature(label),
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_raw': _bytes_feature(img_raw)}))
    
    writer.write(example.SerializeToString())

writer.close()