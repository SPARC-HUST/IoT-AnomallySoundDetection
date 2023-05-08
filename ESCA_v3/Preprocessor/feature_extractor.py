
import numpy as np
import os, sys
sys.path.append(os.getcwd())
from gammatone_filter import gtgram
from config import autocfg as cfg
import librosa
from Preprocessor import TF_WRITER
from multiprocessing import Pool
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class SpectrogramExtractor(TF_WRITER):
    """SpectrogramExtractor extracts log spectrograms (in dB) from a
    time-series signal.
    ------------------------
    :params: filter_cfg: list augument for filter

    Examples: GAMMATONE_CFG = (window_time, hop_time, channels, f_min)
              LOG_CFG = (frame_rate, hop_time)
    """

    def __init__(self, mode = "from_wav", filter_type='gammatone', filter_cfg=None, samplePerTfrecord=200):
        super(SpectrogramExtractor, self).__init__(mode, samplePerTfrecord)

        self.filter_type = filter_type
        self.filter_cfg  = filter_cfg

    def extract(self, signal):
        # Switching filter
        if self.filter_type == 'gammatone':
            spectogram = self._get_gamma_feature(signal)
        elif self.filter_type == 'mel':
            spectogram = self._get_mel_feature(signal)
        elif self.filter_type == None:
            spectogram = self._get_log_feature(signal)
        return spectogram
    
    def _get_gamma_feature(self, signal):
        chunk = np.array(signal.get_array_of_samples(), dtype=np.float32)/(2*(8*signal.sample_width-1)+1)
        gtg = gtgram.gtgram(chunk, signal.frame_rate, self.filter_cfg)
        return np.flipud(20 * np.log10(gtg))
    
    def _get_mel_feature(self, signal):
        """Not used yet
        ---Incomplete
        """
        pass

    def _get_log_feature(self, signal):
        frame_rate, hop_length = self.filter_cfg
        stft = librosa.stft(signal,
                            n_fft=frame_rate,
                            hop_length=hop_length)[:-1]
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram
      

# if __name__ == '__main__':
#     featureExtractor = SpectrogramExtractor(filter_cfg=cfg.GAMMATONE_SETTING)
#     featureExtractor.export_tfrecord()

