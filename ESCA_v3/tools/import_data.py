
from os.path import join, isdir, normpath
from pydub import AudioSegment
from pydub.utils import make_chunks
from tqdm import tqdm
import os, sys
sys.path.append(os.getcwd())
from multiprocessing import Pool
from config import autocfg as cfg
from config.autocfg import create_folder, get_name
import glob

def check_filesize(filePath, size=176444):
    if os.path.getsize(filePath) != size:
        return False
    else: return True
def remove_file(filePath):
    os.remove(filePath)

def remove_illegal_file(filePath):
    if check_filesize(filePath):
        pass
    else: remove_file(filePath)

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
    file_list = cfg.get_list_dir(src)
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

    sourcePath = cfg.DATA_SOURCE
    folderList = cfg.get_list_dir(sourcePath)
    for folder in folderList:
        storageSubolder = normpath(cfg.BASE_DATA_PATH)
        destinationPath = join(cfg.DATA_PATH['raw'], get_name(folder))
        split_data(folder, destinationPath, 2)
    print('Data is imported complete!')


    filePathList = [f for f in glob.glob(cfg.DATA_PATH['raw'] + '/**/*.wav')]
    pool = Pool(processes=2)
    with tqdm(desc="Illigal Checking :", total=len(filePathList),\
            bar_format ='{desc:<15}{percentage:3.0f}%|{bar:50}{r_bar}') as pbar:
        for _ in pool.map(remove_illegal_file, filePathList):      # process imagePathList iterable with pool
            pbar.update()

    illigalFileList = [f for f in glob.glob(cfg.DATA_PATH['raw'] + '/**/*.wav')]
    print('Number of inllgal files: ', len(filePathList)-len(illigalFileList))