import os, sys
sys.path.append(os.getcwd())
from config import autocfg as cfg

from multiprocessing import Pool
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from Preprocessor import SpectrogramExtractor


featureExtractor = SpectrogramExtractor(filter_cfg=cfg.GAMMATONE_SETTING)
featureExtractor.export_tfrecord()