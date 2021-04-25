import tensorflow as tf
import argparse
import yaml
import numpy as np
from pathlib import Path

from nets.simple_baseline import SimpleBaseline
from nets.efficient_net_lite import EfficientNetLite
from lr_schedules import WarmupCosineDecay
from utils.dictToObject import DictToObject
import cost_functions as cf

def create_model(cfg):
  if cfg.MODEL.NAME == 'SimpleBaseline':
    model = SimpleBaseline(cfg.DATASET.INPUT_SHAPE)
  elif cfg.MODEL.NAME == 'EfficientNetLite':
    model = EfficientNetLite(cfg.MODEL.SIZE)

  
  lr = cfg.TRAIN.LR * cfg.TRAIN.BATCH_SIZE / 32

  lr_schedule = WarmupCosineDecay(
              initial_learning_rate=lr,
              decay_steps=cfg.TRAIN.EPOCHS * cfg.SPE,
              warmup_steps=cfg.TRAIN.WARMUP_EPOCHS * cfg.SPE,
              warmup_factor=cfg.TRAIN.WARMUP_FACTOR)
              
  model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule), loss=cf.mse)
  return model

def get_config(name='local'): 
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', default=f"./configs/{name}.yaml")
  parser.add_argument('-t', '--tpu', default=False)
  args, unknown = parser.parse_known_args()
  cfg = DictToObject(yaml.safe_load(open(args.cfg)))
  cfg.TPU = args.tpu
  cfg.SPE = int(np.ceil(cfg.DATASET.TRAIN_SIZE / cfg.TRAIN.BATCH_SIZE))
  cfg.VAL_SPE = int(np.ceil(cfg.DATASET.VAL_SIZE / cfg.VAL.BATCH_SIZE))

  Path(f"./models/{cfg.MODEL.SAVE_NAME}").mkdir(parents=True, exist_ok=True)

  return cfg

def get_strategy(tpu):
  if tpu:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))

    strategy = tf.distribute.TPUStrategy(resolver)
  else:
    strategy = None
  
  return strategy