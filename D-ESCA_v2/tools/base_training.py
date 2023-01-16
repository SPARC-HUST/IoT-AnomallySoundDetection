import os
import sys
sys.path.append(os.getcwd())
import tensorflow as tf
import numpy as np
from core.DataLoader import Dataloader
from core.Trainer import ModelTrainer
from helper.parser import arg_parser 
from config import update_config, get_cfg_defaults

gpus = tf.config.list_physical_devices('GPU')

if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024*3.5)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

# if __name__ == '__main__':
#   # merge config from yaml file
#   cfg = get_cfg_defaults()
#   config_file = arg_parser('Create Dataloader for further uses.')
#   cfg = update_config(cfg, config_file)

#   # make a dataloader
#   data_loader = Dataloader(cfg)
#   data_dict = {
#     'train': data_loader.create_dataloader('train'),
#     'test': data_loader.create_dataloader('test'),
#     'val': data_loader.create_dataloader('val'),
#   }

#   # define a trainer
#   base_trainer = ModelTrainer(cfg)
#   # compile and trainer
#   base_trainer.compile()
#   base_trainer.fit(data_dict)

def base_training():
  # merge config from yaml file
  cfg = get_cfg_defaults()
  config_file = arg_parser('Create Dataloader for further uses.')
  cfg = update_config(cfg, config_file)

  # make a dataloader
  data_loader = Dataloader(cfg)
  data_dict = {
    'train': data_loader.create_dataloader('train'),
    'test': data_loader.create_dataloader('test'),
    'val': data_loader.create_dataloader('val'),
  }

  # define a trainer
  base_trainer = ModelTrainer(cfg)
  # compile and trainer
  base_trainer.compile()
  base_trainer.fit(data_dict)

  return 0

base_training()