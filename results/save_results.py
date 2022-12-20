import os 
import sys

sys.path.append('../unet-tf2')

import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision, Recall
from unet_tf2.utils import dice_coef
import argparse

import cv2

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd


def parse_arguments():
    """Read arguments from a command line."""
    parser = argparse.ArgumentParser(description='Save results ablation study')

    parser.add_argument(
        '--mode',
        metavar='MODE',
        type=str,
        default='detection',
        help='mode of ablation settings')

    parser.add_argument(
        '--runid',
        metavar='RUN_ID',
        type=int,
        default=None,
        help='croos validation run id')

    args = parser.parse_args()

    return args


def main(args):

    ground_truth_path = f'./ground_truth/run{args.runid}'
    y_true = tf.keras.utils.image_dataset_from_directory(
        ground_truth_path, labels=None, shuffle=False)
    y_true = tfds.as_numpy(y_true)
    y_true = np.concatenate([x for x in y_true], axis=0)/255

    precision = Precision()
    recall = Recall()

    data_path = os.path.join(".", args.mode, f'run{args.runid}')

    y_pred = tf.keras.utils.image_dataset_from_directory(
        data_path, labels=None, shuffle=False)
    y_pred = tfds.as_numpy(y_pred)
    y_pred = np.concatenate([x for x in y_pred], axis=0)/255

    dice = dice_coef(y_true, y_pred).numpy()
    prec = precision(y_true, y_pred).numpy()
    recll = recall(y_true, y_pred).numpy()

    row = [args.runid, dice, prec, recll]
    print(row)
    line = ",".join([str(x) for x in row]) + "\n"

    result_path = os.path.join(".", "results", f"{args.mode}.txt")

    if not os.path.exists(result_path):
        with open(result_path, 'x') as f:
            line = "run_id,dice,precision,recall\n" + line
            f.write(line)
    else:
        with open(result_path, "a") as f:
            f.write(line)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
