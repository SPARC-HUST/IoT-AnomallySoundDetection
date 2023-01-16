import tensorflow as tf
import numpy as np
import os

class Preprocessor():
    def __init__(self, cfg) -> None:
        self.stat_file = cfg.DATASET.PATH.TFRECORDS
        self._load_statistic_npz()

    def add_dimentsion(self, input):
        return tf.expand_dims(input, axis=-1)

    def rescale(self, input):
        return tf.clip_by_value(t=(input-self.min)/self.denominator, clip_value_min=0.0, clip_value_max=1.0)

    def _load_statistic_npz(self):
        with open(os.path.join(self.stat_file, 'stats.npz'), 'rb') as file:
            holder = np.load(file)
            self.max = holder['max']
            self.min = holder['min']
        self.denominator = self.max - self.min
    
    def get_max_min(self):
        with open(os.path.join(self.stat_file, 'stats.npz'), 'rb') as file:
            holder = np.load(file)
            self.max = holder['max']
            self.min = holder['min']
        return self.max, self.min