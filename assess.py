#%%
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import create_model, get_strategy, get_config
import utils.plots as pl
import utils.predictions as pu
from utils.const import NAMES, SIGMA
from tfds_loader import load_plain_ds, load_ds

def assess_oks(dataset, model, name):
    headers = ['img_id', 'OKS']
    headers.extend(NAMES)
    rows = []
    skipped = 0
    for batch in val_dataset:
        batch_pred_hm = model.predict(batch[1])
        for i, pred_hm in enumerate(batch_pred_hm):
            # img_id, img, height, width, areas, kp, hm
            img_id = batch[0][i]
            height = batch[2][i]
            width = batch[3][i]
            area = batch[4][i][0]
            kp_gt = batch[5][i][0]  # first person

            if np.sum(batch[5][i][1][:, -1]) != 0:
                skipped += 1
                continue

            pred_kp = pu.get_preds(pred_hm, (height, width))

            visible = kp_gt[:, -1] > 0
            visibility = tf.cast(visible, dtype=tf.float64)

            gt = tf.cast(kp_gt[:, 0:2], tf.float64)
            squared = (pred_kp - gt)**2
            d_square = tf.math.reduce_sum(squared, axis=1)
            area_square = area ** 2
            divider = tf.cast(SIGMA**2 * 2 * area, dtype=tf.float64)
            exps_inside = -1 * d_square/divider
            exps = tf.math.exp(exps_inside)
            exps *= visibility
            exps_sum = tf.math.reduce_sum(exps)
            visible_num = tf.math.reduce_sum(visibility)
            oks = exps_sum/visible_num

            elements_exps = exps - tf.cast(tf.math.logical_not(visible),dtype=tf.float64)  # -1 if not visible
            row = [img_id.numpy(), oks.numpy()]
            row.extend(elements_exps.numpy())
            rows.append(row)

    df = pd.DataFrame(rows,columns=headers)
    df = df.replace(-1, np.nan)
    
    return df

def calculate_ap(column, ap):
        count = column.shape[0]
        return (column >= ap).sum()/count

def calculate_aps_dataframe(oks_df):
    prec_headers = ["AP", "OKS"]

    prec_headers.extend(NAMES)

    ap_df= pd.DataFrame(columns=prec_headers)
    aps = np.linspace(0, .95, 20)
    aps_strings = ['mAP=(0.5-0.95)']
    aps_strings.extend(["AP=%.2f" % ap for ap in aps])
    ap_df["AP"] = aps_strings

    mAP_values = np.linspace(.5, 0.95, 10)
    for i, column in enumerate(oks_df):
        if i == 0:
            continue
        valid = oks_df[column].dropna()
        ap_values = [ calculate_ap(valid, ap) for ap in aps ]
        mAPs_steps = ap_values[10:]
        mAP = sum(mAPs_steps)/len(mAPs_steps)
        col = [mAP]
        col.extend(ap_values)
        ap_df[column] = col
    
    return ap_df

def assess_dataset(dataset, model, name):
    oks_df = assess_oks(dataset, model, name)
    oks_df.to_csv(f"./models/{cfg.MODEL.SAVE_NAME}/{name}_oks.csv", index=False)

    ap_df = calculate_aps_dataframe(oks_df)
    ap_df.to_csv(f"./models/{cfg.MODEL.SAVE_NAME}/{name}_aps.csv", index=False)
    

def display_training_curves(training, validation, title):
        fig, ax = plt.subplots()
        plt.plot(training)
        plt.plot(validation)
        plt.title('model '+ title)
        plt.ylabel(title)
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'])
        plt.savefig("test.svg")
        return plt


def draw_history(cfg):
    with open(f'./models/{cfg.MODEL.SAVE_NAME}/training.history', "rb") as f:
        history = pickle.load(f)
    plt = display_training_curves(history['loss'], history['val_loss'], 'loss')
    plt.savefig(f'./models/{cfg.MODEL.SAVE_NAME}/loss.svg')

    
#%%
cfg = get_config()
strategy = get_strategy(cfg.TPU)

train_dataset = load_plain_ds(cfg.DATASET.TRAIN_DIR, cfg.TRAIN.BATCH_SIZE, cfg.DATASET.INPUT_SHAPE, cfg.DATASET.OUTPUT_SHAPE)
val_dataset = load_plain_ds(cfg.DATASET.VAL_DIR, cfg.VAL.BATCH_SIZE,
                            cfg.DATASET.INPUT_SHAPE, cfg.DATASET.OUTPUT_SHAPE)

if strategy != None:
    with strategy.scope():
        model = create_model(cfg)
        model.load_weights(
            f'./models/{cfg.MODEL.SAVE_NAME}/model.h5', by_name=True)
else:
    model = create_model(cfg)
    model.load_weights(f'./models/{cfg.MODEL.SAVE_NAME}/model.h5', by_name=True)
#%%
assess_dataset(val_dataset, model, "val")
assess_dataset(train_dataset, model, "train")
draw_history(cfg)
