import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
AUTOTUNE = tf.data.AUTOTUNE
GCS_PATH = "gs://kds-b38ce1b823c3ae623f5691483dbaa0f0363f04b0d6a90b63cf69946e"
BATCH_SIZE = 64
IMAGE_SIZE = [1024, 1024]

def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def read_tfrecord(example, labeled):
    tfrecord_format = (
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
        if labeled
        else {"image": tf.io.FixedLenFeature([], tf.string),}
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return image, label
    return image


filenames = ['./Data/tfrecord/Target3/normal_001.tfrecord']
raw_dataset = tf.data.TFRecordDataset(filenames)
print(raw_dataset)

# Create a description of the features.
feature_description = {
    'feature': tf.io.FixedLenFeature([], tf.float32),
    'label': tf.io.FixedLenFeature([], tf.string),
    'idx': tf.io.FixedLenFeature([], tf.string),
}
def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)

parsed_dataset = raw_dataset.map(_parse_function)
print(parsed_dataset)

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)

# result = {}
# # example.features.feature is the dictionary
# for key, feature in example.features.feature.items():
#   # The values are the Feature objects which contain a `kind` which contains:
#   # one of three fields: bytes_list, float_list, int64_list

#   kind = feature.WhichOneof('kind')
#   result[key] = np.array(getattr(feature, kind).value)

# print(result)