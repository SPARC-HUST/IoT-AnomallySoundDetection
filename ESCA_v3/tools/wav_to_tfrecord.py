import tensorflow as tf


from os.path import join, isdir, normpath
from pydub import AudioSegment
from pydub.utils import make_chunks
from tqdm import tqdm
import os, sys
sys.path.append(os.getcwd())
from multiprocessing import Pool
from config import autocfg as cfg
import glob
import numpy as np
from gammatone import gtgram
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _get_gamma_feature(input):
    chunk = np.array(input.get_array_of_samples(), dtype=np.float32)/(2*(8*input.sample_width-1)+1)
    gtg = gtgram.gtgram(chunk, fs= input.frame_rate, window_time= 0.06*2, \
        hop_time = 0.06, channels = 32, f_min=100)
    return np.flipud(20 * np.log10(gtg))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    value = tf.io.serialize_tensor(value)
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def serialize_feature(feature, label, id):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    feature = np.reshape(feature, [32*32,])
    featurePackage = {
        'feature': _float_feature(feature),
        'label': _bytes_feature(label),
        'idx': _bytes_feature(id)
    }
    # Create a Features message using tf.train.Example.
    temp_proto = tf.train.Example(features=tf.train.Features(feature=featurePackage))
    return temp_proto.SerializeToString()


def _get_label_by_path(filePath):
    if cfg.get_folder_name(filePath) == 'normal':
       label = 0
    elif cfg.get_folder_name(filePath) == 'abnormal':
       label = 1
    else:
       raise ValueError("Data path is specified in subfolders 'normal' and 'abnormal'")
    return label


def _write_tfRecord(featurePackage, filePath):
    cfg.create_folder(cfg.get_folder_path(filePath))
    featureList = featurePackage['feature']
    labelList   = featurePackage['label']
    idxList     = featurePackage['idx']


    with tf.io.TFRecordWriter(filePath) as writer:
        for feature, label, idx in zip(featureList, labelList, idxList):
            temp_proto = serialize_feature(feature, label, idx)
            writer.write(temp_proto)



dataType = 'normal'
filePathList = [f for f in glob.glob(cfg.DATA_PATH['raw'] + f'/{dataType}/*.wav')]

featureList = []; idxList = []; labelList = []; fileCounter = 0

for filePath in tqdm(filePathList, desc=f'Extracting features'):

    file = AudioSegment.from_file(filePath, "wav")

    featureList.append(_get_gamma_feature(file)),
    labelList.append(_get_label_by_path(filePath)),
    idxList.append(cfg.get_name(filePath))

    featurePackage = {
        'feature'   : featureList,
        'label'     : labelList,
        'idx'       : idxList
    }
    if len(featureList) == 200:
        fileCounter += 1
        filename = '{}_{:03d}.tfrecord'.format(dataType, fileCounter)
        tfrecordPath = join(cfg.DATA_PATH['tfrec'], filename)
        _write_tfRecord(featurePackage, tfrecordPath)

        #reset list and counter
        featureList = []; idxList = []; labelList = []; sampleCounter = 0
    else:
       pass

    







    