import os
import sys
sys.path.append(os.getcwd())
import tensorflow as tf
import numpy as np
from core.DataLoader import Dataloader
from core.Trainer import TL_Trainer
from helper.parser import arg_parser 
from config import update_config, get_cfg_defaults

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024*3)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

def load_all_files_from_dirs(list_of_dirs):
  list_of_files = []
  for dir in list_of_dirs:
    files = os.listdir(dir)
    for f in files:
      list_of_files.append(os.path.join(dir, f))
  return list_of_files

if __name__ == '__main__':
  # merge config from yaml file
  cfg = get_cfg_defaults()
  config_file = arg_parser('Create Dataloader for further uses.')
  cfg = update_config(cfg, config_file)

  # prepare directories (can be integrated to Dataloader)
  # normal_files = load_all_files_from_dirs(cfg.TRANSFER_LEARNING.NORMAL_DATA_DIRS)
  # anomaly_file = load_all_files_from_dirs(cfg.TRANSFER_LEARNING.ANOMALY_DATA_DIRS)
  # make a dataloader
  data_loader = Dataloader(cfg)
  data_dict = {
    # 'train': data_loader.create_dataloader_from_files(normal_files),
    'train': data_loader.create_dataloader('normal_tl'),
    'test': data_loader.create_dataloader('test'),
    'val': data_loader.create_dataloader('val'),
  }
  # training_anomaly = data_loader.create_dataloader_from_files(anomaly_file)
  training_anomaly = data_loader.create_dataloader('anomaly_tl')
  # define a trainer
  base_trainer = TL_Trainer(cfg, training_anomaly=training_anomaly)
  # compile and trainer
  base_trainer.compile()
  base_trainer.fit(data_dict)
