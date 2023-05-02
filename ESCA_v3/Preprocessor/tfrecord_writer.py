from os.path import join
from pydub import AudioSegment
import numpy as np
import os, sys
sys.path.append(os.getcwd())
from config import autocfg as cfg
import glob
import tensorflow as tf
from tqdm import tqdm

class TF_WRITER():
    def __init__(self, mode, samplePerTFfile):
        # audio directory and output directory

        self.mode = mode
        self.samplePerTFfile = samplePerTFfile

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

    # Main-process
    def export_tfrecord(self):
        dataTypes = ['normal','abnormal']
        for dataType in dataTypes:
            filePathList = [f for f in glob.glob(cfg.DATA_PATH['raw'] + f'/{dataType}/*.wav')]

            featureList = []; idxList = []; labelList = []; fileCounter = 0

            for filePath in tqdm(filePathList, desc=f'Extracting features'):

                file = AudioSegment.from_file(filePath, "wav")

                featureList.append(self._get_gamma_feature(file)),
                labelList.append(self._get_label_by_path(filePath)),
                idxList.append(cfg.get_name(filePath))

                featurePackage = {
                    'feature'   : featureList,
                    'label'     : labelList,
                    'idx'       : idxList
                }
                if len(featureList) == self.samplePerTFfile:
                    fileCounter += 1
                    filename = '{}_{:03d}.tfrecord'.format(dataType, fileCounter)
                    tfrecordPath = join(cfg.DATA_PATH['tfrec'], filename)
                    self._write_tfRecord(featurePackage, tfrecordPath)

                    #reset list and counter
                    featureList = []; idxList = []; labelList = []; sampleCounter = 0
                else:
                    pass


    def _get_gamma_feature(self, input):
        raise NotImplementedError
    
    def _get_mel_feature(self, input):
        raise NotImplementedError
    

