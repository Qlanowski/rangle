import tensorflow as tf
import pickle
import numpy as np
import pandas as pd

from common import create_model, get_strategy, get_config
import utils.plots as pl
import utils.predictions as pu
from utils.const import NAMES, SIGMA
from tfds_loader import load_plain_ds, load_ds

if __name__ == "__main__":
    cfg = get_config()
    strategy = get_strategy(cfg.TPU)

    # train_dataset = load_plain_ds(cfg.DATASET.TRAIN_DIR, cfg.TRAIN.BATCH_SIZE, cfg.DATASET.INPUT_SHAPE, cfg.DATASET.OUTPUT_SHAPE)
    val_dataset = load_plain_ds(cfg.DATASET.VAL_DIR, cfg.VAL.BATCH_SIZE,
                                cfg.DATASET.INPUT_SHAPE, cfg.DATASET.OUTPUT_SHAPE)

    if strategy != None:
        with strategy.scope():
            model = create_model(cfg)
            model.load_weights(
                f'./models/{cfg.MODEL.SAVE_NAME}.h5', by_name=True)
    else:
        model = create_model(cfg)
        model.load_weights(f'./models/{cfg.MODEL.SAVE_NAME}.h5', by_name=True)

    headers = ['img_id', 'OKS']
    headers.extend(NAMES)
    df = pd.DataFrame(columns=headers)
    skipped = 0
    for batch in val_dataset:
        batch_pred_hm = model.predict(batch[1])
        for i, pred_hm in enumerate(batch_pred_hm):
            # img_id, img, height, width, areas, kp, hm
            img_id = batch[0][i].numpy()
            height = batch[2][i].numpy()
            width = batch[3][i].numpy()
            area = batch[4][i][0].numpy()
            kp_gt = batch[5][i][0]  # first person

            if np.sum(batch[5][i][1][:, -1]) != 0:
                skipped += 1
                continue

            pred_kp = pu.get_preds(pred_hm, (height, width))
            visibility = kp_gt[:, -1] > 0

            gt = kp_gt[:, 0:2]
            squared = (pred_kp - gt)**2
            d_square = np.sum(squared, axis=1)
            d_square *= visibility
            area_square = area ** 2
            divider = SIGMA**2 * 2 * area_square
            exp_inside = -1 * d_square/divider
            visible_num = np.sum(visibility)
            sum_exp = np.sum(exp_inside)
            oks = np.exp(sum_exp)/visible_num

            elements_exps = exp_inside - np.invert(visibility) #-1 if not visible
            row = [img_id,oks]
            row.expand(elements_exps)

            df.append(pd.DataFrame(row).T)
