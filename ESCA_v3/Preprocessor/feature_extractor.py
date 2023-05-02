
import numpy as np
from gammatone_filter import gtgram
import os, sys
sys.path.append(os.getcwd())
from Preprocessor import TF_WRITER
import json

import tensorflow as tf

class Feature_extractor(TF_WRITER):
    def __init__(self, input=None, dst=None, mode='from_wav', filterType=None, filterCfg=None, samplePerTFfile = 200):
        super(Feature_extractor, self).__init__(mode, samplePerTFfile)
        # audio directory and output directory
        self.input = input
        self.dst = dst
        
        # usage of feature extractor: to prepare data from audio files (from_file) or from streaming data (real_time)
        self.mode = mode
        self.type = type

        # length of each audio file
        self.samplePerTFfile = samplePerTFfile 

        # parameters for gamma features/mel features
        self.filterCfg = filterCfg
        
        # if filterType == 'gammatone':
        #     self.filterCfg = filterCfg
        # elif filterType == 'mel_band':
        #     self.filterCfg = filterCfg

        # parameters for mel features

        
        # a dictionary to save back all ids
        self.data = {}
    
    def _get_gamma_feature(self, input):

        chunk = np.array(input.get_array_of_samples(), dtype=np.float32)/(2*(8*input.sample_width-1)+1)
        gtg = gtgram.gtgram(chunk, input.frame_rate, self.filterCfg)
        return np.flipud(20 * np.log10(gtg))
    
    def _get_mel_feature(self, input):
        pass

