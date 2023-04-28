import tensorflow as tf
from os.path import join
from pydub import AudioSegment
from tqdm import tqdm
import glob
import numpy as np
from config import autocfg as cfg

import os, sys
sys.path.append(os.getcwd())
from multiprocessing import Pool
from gammatone_filter import gtgram
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def _get_gamma_feature(input):
    # .reshape((-1, item.channels)) if needed
    chunk = np.array(input.get_array_of_samples(), dtype=np.float32)/(2*(8*input.sample_width-1)+1)
    gtg = gtgram.gtgram(chunk, input.frame_rate,cfg.GAMMATONE_SETTING)
    return np.flipud(20 * np.log10(gtg))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""    
    value= tf.io.serialize_tensor(value)
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def serialize_feature(feature, label, id):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    featurePackage = {
        'feature': _bytes_feature(feature.astype(np.float32)),
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

    







    