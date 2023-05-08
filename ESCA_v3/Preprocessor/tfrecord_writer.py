from os.path import join
from pydub import AudioSegment
import numpy as np
import os, sys
sys.path.append(os.getcwd())
from config import autocfg as cfg
from Preprocessor import MinMaxNormaliser
import glob
import tensorflow as tf
from tqdm import tqdm
import pickle


min_max_normaliser = MinMaxNormaliser(0, 1)

class TF_WRITER():
    def __init__(self, mode, sample_per_tfrecord):
        # audio directory and output directory

        self.mode = mode
        self.sample_per_tfrecord = sample_per_tfrecord
        self.min_max_values = {}

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""    
        # value= tf.io.serialize_tensor(value)
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

        feature = np.reshape(feature, [32*32,])
        featurePackage = {
            # 'feature'   : self._float_feature(feature),
            'feature'   : self._float_feature(feature.astype(np.float32)),
            'label'     : self._int64_feature(label),
            'idx'       : self._bytes_feature(id.encode('utf-8'))
        }
        # Create a Features message using tf.train.Example.
        temp_proto = tf.train.Example(features=tf.train.Features(feature=featurePackage))
        return temp_proto.SerializeToString()
    
    def _get_label_by_path(self, filePath):
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

    def save_min_max_values(self, saved_dir, min_max_values):
        save_path = os.path.join(saved_dir,
                                 "min_max_values.pkl")
        self._save(min_max_values, save_path)

    def _store_min_max_value(self, id, min_val, max_val):
        self.min_max_values[id] = {
            "min": min_val,
            "max": max_val
        }

    # Main-process
    def export_tfrecord(self, data_dir=None):
        """Collecting extracted feature from wav, nomormlize them
        and put them into .tfrecord files
        1- load .wav
        2- get Spectrogram feature
        3- normalize spectrogram feature
        4- save to .tfrecord file"""
        dataTypes = ['normal','abnormal']
        for dataType in dataTypes:
            filePathList = [f for f in glob.glob(cfg.DATA_PATH['raw'] + f'/{dataType}/*.wav')]

            featureList = []; idxList = []; labelList = []; fileCounter = 0

            for filePath in tqdm(filePathList, desc=f'Extracting features', \
                                 bar_format='{desc:<15}{percentage:3.0f}%|{bar:50}{r_bar}'):
                #step 1
                file = AudioSegment.from_file(filePath, "wav")
                #step 2- get Spectrogram feature
                feature = self._get_gamma_feature(file)
                #step 3- normalize spectrogram feature
                norm_feature = min_max_normaliser._normalize(feature)
                #append feature to list for saving tfrecord
                featureList.append(norm_feature)
                labelList.append(self._get_label_by_path(filePath))
                idxList.append(cfg.get_name(filePath))
                #Store min-max values of feature
                self._store_min_max_value(cfg.get_name(filePath),
                                          feature.min(),
                                          feature.max())
                featurePackage = {
                    'feature'   : featureList,
                    'label'     : labelList,
                    'idx'       : idxList
                }
                #step 4- save to .tfrecord file"
                if len(featureList) == self.sample_per_tfrecord \
                    or (len(featureList) == len(filePathList) - fileCounter*self.sample_per_tfrecord):
                    fileCounter += 1
                    filename = '{}_{:03d}.tfrecord'.format(dataType, fileCounter)
                    tfrecordPath = join(cfg.DATA_PATH['tfrec'], filename)
                    self._write_tfRecord(featurePackage, tfrecordPath)

                    #reset list and counter
                    featureList = []; idxList = []; labelList = []; sampleCounter = 0
                else:
                    pass
        self.save_min_max_values(cfg.DATA_PATH['raw'],  self.min_max_values)

    def _get_gamma_feature(self, input):
        raise NotImplementedError
    
    def _get_mel_feature(self, input):
        raise NotImplementedError

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

