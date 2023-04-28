###This is the optimized part of the code

"""
Autocfg.py helps to setup a path with the given dataset and implementation method
-------
None
"""

import os, sys
sys.path.append(os.getcwd())

from config import update_config, get_cfg_defaults

cfg = get_cfg_defaults()
cfg = update_config(cfg, './config/params.yaml')

#################################################################
def get_name(path):
    path = os.path.normpath(path)
    name = os.path.basename(path)
    return name

def get_folder_path(path, depth=1):
    folderPath = path
    for _ in range(depth):
        folderPath = os.path.split(os.path.abspath(folderPath))[0]
    return folderPath

def get_folder_name(path):
    '''Get folder name from path
    ----------
    '''
    path = os.path.normpath(path)
    folderPath = os.path.split(path)[0]
    name = os.path.basename(folderPath)

    return name


def create_folder(path, folderName= ''):
    path = os.path.normpath(path)
    folderPath = os.path.join(path, folderName)
    if os.path.exists(folderPath):
        pass
    else:
        os.makedirs(folderPath)
    return 0



def switch_result_path(impMethod):
    if impMethod == 'base':
        RESULTS = os.path.join(cfg.RESULTS, 'base_training')
        create_folder(RESULTS, '')
    elif impMethod == 'transfer':
        RESULTS = os.path.join(cfg.RESULTS, 'transfer_training')
        create_folder(RESULTS, '')
    elif impMethod == 'realtime':
        RESULTS = os.path.join(cfg.RESULTS, 'Realtime_imp')
        create_folder(RESULTS, '')
    else:
        print('Please give a implementation "base" or "transfer" or "realtime"')
        return
    return RESULTS


#################################################################
BASE_DATA_PATH = cfg.DATA.PATH.BASEPATH
DATA_SOURCE = cfg.DATA.SOURCE
DATASET_NAME = get_name(DATA_SOURCE)
# init_data_path(BASE_DATA_PATH)

DATA_PATH = {
    "raw": os.path.join(BASE_DATA_PATH, 'raw', DATASET_NAME),
    "normal": os.path.join(BASE_DATA_PATH, 'raw', DATASET_NAME, 'normal'),
    "abnormal": os.path.join(BASE_DATA_PATH, 'raw', DATASET_NAME,'abnormal'),
    "tfrec": os.path.join(BASE_DATA_PATH, 'tfrecord', DATASET_NAME)
}

TRAIN_CFG = {
    "epoch": cfg.TRAINING.EPOCH,
    "lr": cfg.TRAINING.LEARNING_RATE,
    "num_worker": None,
    "batchsize": None
}
window_time, hop_time, channels, f_min = 0.06*2, 0.06, 32, 100
GAMMATONE_SETTING = (window_time, hop_time, channels, f_min)

    


    



