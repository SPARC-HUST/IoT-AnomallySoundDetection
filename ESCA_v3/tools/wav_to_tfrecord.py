import tensorflow as tf


from os.path import join, isdir, normpath
from pydub import AudioSegment
from pydub.utils import make_chunks
from tqdm import tqdm
import os, sys
sys.path.append(os.getcwd())
from multiprocessing import Pool
from config import autocfg
from config.autocfg import create_folder, get_name
import glob
import numpy as np
from gammatone import gtgram


def _get_gamma_feature(self, input):
    # .reshape((-1, item.channels)) if needed
    chunk = np.array(input.get_array_of_samples(), dtype=np.float32)/(2*(8*input.sample_width-1)+1)
    # print(chunk.shape)
    gtg = gtgram.gtgram(chunk, fs= 44100, window_time=, \
        hop_time = 0.06, channels = 32, f_min=100)
    return np.flipud(20 * np.log10(gtg))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_feature(feature, label, id):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
   
    feature = {
        'feature': _bytes_feature(feature),
        'label': _bytes_feature(label),
        'idx': _bytes_feature(id)
    }

    # Create a Features message using tf.train.Example.

    temp_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return temp_proto.SerializeToString()

type = 'normal'
filePathList = [f for f in glob.glob(autocfg.DATA_PATH['raw'] + f'/{type}/*.wav')]



featureList = []; idxList = []; label = []; sampleCounter = 0

for filePath in tqdm(filePathList, desc=f'Extracting features'):

    file = AudioSegment.from_file(filePath, "wav")
    feature = _get_gamma_feature(file)
    featureList.append(feature)




    