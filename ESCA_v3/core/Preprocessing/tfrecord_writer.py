from helper.utils import read_file_name, extract_mbe
from os.path import join, isdir, split
from os import makedirs
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
from gammatone import gtgram
from config import autocfg as cfg

import tensorflow as tf

class TF_WRITER():
    def __init__(self, src=None, dst=None, mode='from_wav'):
        # audio directory and output directory
        self.src = src
        self.dst = dst
        
        # usage of feature extractor: to prepare data from audio files (from_file) or from streaming data (real_time)
        self.mode = mode
        self.type = type

 
        # a dictionary to save back all ids
        self.data = {}

    
    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""    
        value= tf.io.serialize_tensor(value)
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() 
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(self,value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
    def serialize_feature(self, feature, label, id):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """

        # feature = np.reshape(feature, [32*32,])
        featurePackage = {
            # 'feature'   : self._float_feature(feature),
            'feature'   : self._bytes_feature(feature.astype(np.float32)),
            'label'     : self._bytes_feature(label),
            'idx'       : self._bytes_feature(id)
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

    def _write_tfRecord(self, featurePackage, filePath):
        cfg.create_folder(cfg.get_folder_path(filePath))
        featureList = featurePackage['feature']
        labelList   = featurePackage['label']
        idxList     = featurePackage['idx']


        with tf.io.TFRecordWriter(filePath) as writer:
            for feature, label, idx in zip(featureList, labelList, idxList):
                temp_proto = self.serialize_feature(feature, label, idx)
                writer.write(temp_proto)

    

