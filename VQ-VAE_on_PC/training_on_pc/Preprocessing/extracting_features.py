from .utils import read_file_name, extract_mbe
from os.path import join, isdir
from os import mkdir
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
from gammatone import gtgram
import json


class Extract_features():
    def __init__(self):
        self.data = {}

    # a function segment the audio file to deserved length in second
    # UPDATE: MAKE FEATURES DIRECTLY FROM TRUNKS AND SAVE THEM INSTEAD
    def gamma_features(self, src, dst, length, window_time, hop_time, channels, f_min):    # noqa: E501
        # src is the source folder
        # dst is the destination folder

        time_per_file = length*1000  # time is processed in millisecond
        if not isdir(src):
            raise ValueError('Folder does not exist.')
        print(src)
        file_list = read_file_name(src)
        if len(file_list) == 0:
            return 1

        # check if dst is valid
        if not isdir(dst):
            mkdir(dst)

            print(dst)

        rate = 10//length
        file_nums = len(file_list)
        num = (file_nums*rate)//200
        remainder = (file_nums*rate) % 200
        i = 0
        j = 0
        orders = []
        feature = []

        for file in file_list:
            audio = AudioSegment.from_file(file, "wav")
            # print('audiosegment')
            chunks = make_chunks(audio, time_per_file)
            # print('chunking')

            # write chunks to dst
            for index, item in enumerate(chunks):
                # .reshape((-1, item.channels)) if needed
                chunk = np.array(item.get_array_of_samples(), dtype=np.float32)/(2*(8*item.sample_width-1)+1)   # noqa: E501
                # print(chunk.shape)
                gtg = gtgram.gtgram(chunk, item.frame_rate, window_time, hop_time, channels, f_min)    # noqa: E501
                a = np.flipud(20 * np.log10(gtg))
                if (a.shape != (32, 32)):
                    remainder -= 1
                    if remainder < 0:
                        remainder = 199
                        num -= 1
                    # os.remove(file)
                    continue
                # print(a.shape)
                feature.append(a)
                name = file.split('/')[-1]
                orders.append(name[:-4]+'_'+str(index)+'.wav')
                j += 1

                if (j % 200 == 0) or (i == num and j == remainder) or (num == 0 and j == remainder):    # noqa: E501
                    normal_feature = np.array(feature)
                    print(i)
                    print(normal_feature.shape)
                    np.savez_compressed(join(dst, str(i)), feature)
                    i += 1
                    j = 0
                    feature = []


        name_list = dst.split('/')
        category = name_list[-2]+'_'+name_list[-1]
        print(f'Saving {category}')
        self.data[category] = orders

        return 0

    def mel_features(self, src, dst, length, sr, nfft, nb_mel_bands):    # noqa: E501
        time_per_file = length*1000  # time is processed in millisecond

        print(src)
        if not isdir(src):
            raise ValueError('Folder does not exist.')

        file_list = read_file_name(src)
        if len(file_list) == 0:
            return 1

        print(dst)
        # check if dst is valid
        if not isdir(dst):
            mkdir(dst)

        rate = 10//length
        file_nums = len(file_list)
        num = (file_nums*rate)//200
        remainder = (file_nums*rate) % 200
        i = 0
        j = 0
        orders = []
        feature = []

        for file in file_list:
            audio = AudioSegment.from_file(file, "wav")
            # print('audiosegment')
            chunks = make_chunks(audio, time_per_file)
            # print('chunking')

            # write chunks to dst
            for index, item in enumerate(chunks):
                # .reshape((-1, item.channels)) if needed
                chunk = np.array(item.get_array_of_samples(), dtype=np.float32)/(2*(8*item.sample_width-1)+1)   # noqa: E501
                # print(chunk.shape)
                a = extract_mbe(chunk, sr, nfft, nb_mel_bands)
                if (a.shape != (32, 32)):
                    remainder -= 1
                    if remainder < 0:
                        remainder = 199
                        num -= 1
                    # os.remove(file)
                    continue
                # print(a.shape)
                feature.append(a)
                name = file.split('/')[-1]
                orders.append(name[:-4]+'_'+str(index)+'.wav')
                j += 1

                if (j % 200 == 0) or (i == num and j == remainder) or (num == 0 and j == remainder):    # noqa: E501
                    normal_feature = np.array(feature)
                    print(i)
                    print(normal_feature.shape)
                    np.savez_compressed(join(dst, str(i)), feature)
                    i += 1
                    j = 0
                    feature = []


        name_list = dst.split('/')
        category = name_list[-2]+'_'+name_list[-1]
        print(f'Saving {category}')
        self.data[category] = orders
        return 0

    def save_id(self, path, name):
        with open(path + name, 'w') as file:
            json.dump(self.data, file)
        self.data = {}
        return 0
