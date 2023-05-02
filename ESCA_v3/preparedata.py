from config import autocfg as cfg
import os, sys
sys.path.append(os.getcwd())
from multiprocessing import Pool
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from Preprocessor import Feature_extractor


featureExtractor = Feature_extractor(filterCfg=cfg.GAMMATONE_SETTING)
featureExtractor.export_tfrecord()