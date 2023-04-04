from os.path import join, isdir, normpath
from os import mkdir, listdir
from pydub import AudioSegment
from pydub.utils import make_chunks

import os, sys
sys.path.append(os.getcwd())
from config import autocfg
from config.autocfg import create_folder



def get_list_dir(path):
    '''
    a function goes through directoty and gives back list of files
    '''
    file_list = []
    if isdir(path):
        file_list = listdir(path)
        file_list = [join(path, file) for file in file_list]
    else:
        return 0

    return file_list


def split_data(src, dst, length=10):
    '''
    a function segment the audio file to desired length in second
    '''
    # src is the source folder
    # dst is the destination folder
    src = normpath(src)
    dst = normpath(dst)
    timePerFile = length*1000  # time is processed in millisecond

    if not isdir(src):
        return 1
    file_list = get_list_dir(src)
    # print(len(file_list))
    if not isdir(dst):
        os.makedirs(dst)

    for file in file_list:
        audio = AudioSegment.from_file(file, "wav")
        chunks = make_chunks(audio, timePerFile)
        fileName = file.split('\\')[-1]
        print(file, fileName)

        # write chunks to dst
        for index, item in enumerate(chunks):
            chunkName = fileName[:-4] + '_' + str(index) + '.wav'
            # print("exporting ", chunkName)
            item.export(join(dst, chunkName), format='wav')
    return 0

def get_name(path):
    path = normpath(path)
    name = path.split('\\')[-1]
    return name


if __name__ == '__main__':
    # sources and destinations are folders that contain audio files
    # please specify the path to these folder in absolute path

    sourcePath = autocfg.DATA_SOURCE
    folderList = get_list_dir(sourcePath)
    print(folderList)
    for folder in folderList:
        storageSubolder = normpath(autocfg.BASE_DATA_PATH)
        create_folder(storageSubolder, get_name(sourcePath))
        print(storageSubolder, folder)
        # split_data(folder, join(storageSubolder), 2)
    print('Completed')
