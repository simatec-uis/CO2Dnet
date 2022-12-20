import os
import numpy as np
import cv2
import tensorflow as tf


def build_data(x_path, y_path, files_path=None, porcentage=1.0, Xextension='.jpeg', yextension='.png', size = 640):
    
    if isinstance(files_path, list):
        names = files_path
    else:
        with open(files_path) as f:
            names = f.readlines()

    names = [name.replace('\n', '') for name in names]

    Xdata = []
    ydata = []

    for x in names:


        ix = cv2.imread(x_path + '/' + x + Xextension)
        #ix = cv2.resize( ix, (size, size), interpolation = cv2.INTER_AREA)
        
        iy = cv2.imread(y_path + '/' + x + yextension)

        #iy = cv2.resize( iy, (size, size) , interpolation = cv2.INTER_AREA)

        ix = np.asarray(ix)
        iy = np.asarray(iy)

        Xdata.append(ix)
        ydata.append(iy)

    Xdata = np.float32(np.array(Xdata) / 255)
    ydata = np.float32((np.array(ydata) > 10)*1)

    return Xdata, ydata[..., 0, None]


def iou_coef(y_true, y_pred, smooth=1e-5):
    th = 0.5
    y_pred = tf.cast(y_pred > th, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, [1, 2, 3]) + \
        tf.reduce_sum(y_pred, [1, 2, 3])-intersection
    iou = tf.reduce_mean((intersection + smooth) / (union + smooth), axis=0)

    return iou


def dice_coef(y_true, y_pred, smooth=1e-5):
    th = 0.5
    y_pred = tf.cast(y_pred > th, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1, 2, 3])
    denom = tf.reduce_sum(y_true, [1, 2, 3]) + tf.reduce_sum(y_pred, [1, 2, 3])
    dice = tf.reduce_mean((2*intersection + smooth) / (denom + smooth), axis=0)

    return dice
