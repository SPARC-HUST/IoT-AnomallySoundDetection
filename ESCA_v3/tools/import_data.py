
from os.path import join, isdir, normpath
from pydub import AudioSegment
from pydub.utils import make_chunks
from tqdm import tqdm
import os, sys
sys.path.append(os.getcwd())

from config import autocfg
from config.autocfg import create_folder, get_name

def get_list_dir(path):
    '''
    a function goes through directoty and gives back list of files
    '''
    file_list = []
    if isdir(path):
        file_list = os.listdir(path)
        file_list = [join(path, file) for file in file_list]
    else:
        return 0

    return file_list

def split_data(src, dst, length=10):
    '''
    a function segment the audio file to desired length in second
    __________
    src is the source folder
    dst is the destination folder
    '''

    src = normpath(src)
    dst = normpath(dst)
    timePerFile = length*1000  # time is processed in millisecond

    if not isdir(src):
        return 1
    file_list = get_list_dir(src)
    if not isdir(dst):
        os.makedirs(dst)

    for file in tqdm(file_list, desc="Importing",bar_format='{desc:<15}{percentage:3.0f}%|{bar:50}{r_bar}'):
        audio = AudioSegment.from_file(file, "wav")
        chunks = make_chunks(audio, timePerFile)
        fileName = os.path.basename(file)

        # write chunks to dst
        for index, item in enumerate(chunks):
            chunkName = fileName[:-4] + '_' + str(index) + '.wav'
            item.export(join(dst, chunkName), format='wav')
    return 0

if __name__ == '__main__':
    # sources and destinations are folders that contain audio files
    # please specify the path to these folder in absolute path

    sourcePath = autocfg.DATA_SOURCE
    folderList = get_list_dir(sourcePath)
    for folder in folderList:
        storageSubolder = normpath(autocfg.BASE_DATA_PATH)
        split_data(folder, join(autocfg.DATA_PATH['raw'], get_name(folder)), 2)
    print('Data is imported complete!')
