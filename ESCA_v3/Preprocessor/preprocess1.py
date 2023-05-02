from helper.utils import read_file_name, extract_mbe
from os.path import join, isdir, split
from os import makedirs
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
from gammatone import gtgram
import json

import tensorflow as tf

class Feature_extractor():
    def __init__(self, src=None, dst=None, mode='from_wav', type=None,\
        segment_len=None, audio_len=None, sample_per_file=200, \
        window_time=None, hop_time=None, channels=None, f_min=None, \
        sr=16000, nfft=None, n_mel_band=32):
        # audio directory and output directory
        self.src = src
        self.dst = dst
        
        # usage of feature extractor: to prepare data from audio files (from_file) or from streaming data (real_time)
        self.mode = mode
        self.type = type
        # length of each audio file
        self.audio_len = audio_len
        self.segment_len = segment_len
        self.sample_per_file = sample_per_file 
        # parameters for gamma features
        self.window_time = window_time
        self.hop_time = hop_time
        self.channels = channels
        self.f_min = f_min
        # parameters for mel features
        self.sr = sr
        self.nfft = nfft
        self.n_mel_band = n_mel_band

        # a dictionary to save back all ids
        self.data = {}
    def _get_gamma_feature(self, input):
        # .reshape((-1, item.channels)) if needed
        chunk = np.array(input.get_array_of_samples(), dtype=np.float32)/(2*(8*input.sample_width-1)+1)
        gtg = gtgram.gtgram(chunk, fs= input.frame_rate, window_time= 0.06*2, \
            hop_time = 0.06, channels = 32, f_min=100)
        return np.flipud(20 * np.log10(gtg))