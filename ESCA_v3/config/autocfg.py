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
    name = path.split('\\')[-1]
    return name

def create_folder(path, folderName):
    path = os.path.normpath(path)
    folderPath = os.path.join(path, folderName)
    if os.path.exists(folderPath):
        pass
    else:
        os.makedirs(folderPath)
    return 0

def init_data_path(BASE_DATA_PATH):
    ## tree: data---input --[train, val, test]
    #             |--tfrecord-- [train, val, test]
    SUBFOLDERS = ['raw', 'tfrecord_temp']
    FOLDERS = ['train', 'test']
    for subfolder in SUBFOLDERS:
        create_folder(BASE_DATA_PATH, subfolder)
    for folder in FOLDERS:
        create_folder(os.path.join(BASE_DATA_PATH, subfolder), folder)

    return 0        
        
def switch_path(condition):
    # rawTrain = DATA_PATH['raw_train']
    # rawVal = DATA_PATH['raw_val']
    # rawTest = DATA_PATH['raw_test']
    # train = DATA_PATH['train']
    # val = DATA_PATH['val']
    # test = DATA_PATH['test']
    pass

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
init_data_path(BASE_DATA_PATH)

DATA_PATH = {
    "raw": os.path.join(BASE_DATA_PATH, 'raw', DATASET_NAME),
    "normal": os.path.join(BASE_DATA_PATH, 'raw', DATASET_NAME, 'normal'),
    "abnormal": os.path.join(BASE_DATA_PATH, 'raw', DATASET_NAME,'abnormal'),
    "train": os.path.join(BASE_DATA_PATH, 'tfrecord_temp', DATASET_NAME,'train'),
    "test": os.path.join(BASE_DATA_PATH, 'tfrecord_temp', DATASET_NAME,'test')
}

TRAIN_CFG = {
    "epoch": cfg.TRAINING.EPOCH,
    "lr": cfg.TRAINING.LEARNING_RATE,
    "num_worker": None,
    "batchsize": None
}



    


    



