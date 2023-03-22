import os
from os.path import join, isdir, dirname
import sys
sys.path.append(os.getcwd())
from config import update_config, get_cfg_defaults
from helper.parser import arg_parser 
from os.path import join
from os import listdir, getpid, scandir, rename, environ, remove, setpgrp, killpg,_exit
import shutil
import time
def CPfile(input_file, base_file):
    if (os.path.exists(base_file) == False):
        shutil.copy2(input_file, base_file)
cfg = get_cfg_defaults()
config_file = arg_parser('Create Dataloader for further uses.')
cfg = update_config(cfg, config_file)
sample_loc = join(cfg.REALTIME.LOG_PATH,'record')
if not isdir(sample_loc):
    os.makedirs(sample_loc)
data_import = join(cfg.REALTIME.LOG_PATH,'data')
if not isdir(data_import):
    os.makedirs(data_import)
print("Start import audio files")
while True:
    for x in listdir(data_import):
        if x.endswith(".wav"):
            # if 'basefile.wav' in base_file:
            #     continue
            while 'basefile.wav' in listdir(sample_loc):
                pass
            CPfile(join(data_import,x),join(sample_loc,x))
            print('File',x,"have imported !")
            rename(join(sample_loc,x),join(sample_loc,'basefile.wav'))
            os.remove(join(data_import,x))
            
