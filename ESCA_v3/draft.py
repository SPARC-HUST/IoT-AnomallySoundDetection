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
from core.Preprocessing import Feature_extractor


featureExtractor = Feature_extractor(filterCfg=cfg.GAMMATONE_SETTING)
featureExtractor.export_tfrecord()







    