from absl import app, flags, logging
from absl.flags import FLAGS

import os
import json
import tensorflow as tf

from unet_tf2.utils import build_data, iou_coef
from unet_tf2.models.unet import Unet
from tensorflow.keras.metrics import Precision, Recall

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    ModelCheckpoint,
    EarlyStopping)


flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_string(
    'weights_save', './checkpoints/unet_train.tf', 'path to save file')
flags.DEFINE_integer('features', 32, 'features of Unet network')
flags.DEFINE_integer('levels', 5, 'levels of Unet network')
flags.DEFINE_integer('epochs', 10, 'epochs to train')
flags.DEFINE_integer('batch_size', 40, 'batch size')
flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')
flags.DEFINE_enum('mode', 'none', ['none', 'resume'],
                  'none: no load weigths, '
                  'resume: resume pre-training')


def main(_argv):

    x_path = os.path.join(FLAGS.dataset, 'JPEGImages')
    y_path = os.path.join(FLAGS.dataset, 'Mask')
    train_path = os.path.join(FLAGS.dataset, 'ImageSets', 'Main', 'train.txt')
    val_path = os.path.join(FLAGS.dataset, 'ImageSets', 'Main', 'val.txt')

    Xtrain, ytrain = build_data(x_path, y_path, train_path)
    Xval, yval = build_data(x_path, y_path, val_path)

    input_shape = Xtrain.shape[1:]

    model = Unet(input_shape,  FLAGS.features, FLAGS.levels)

    model_name, _ = os.path.splitext(FLAGS.weights_save)
    config_path = model_name + '.txt'
    model_config = model.to_json()

    with open(config_path, "w") as text_file:
        text_file.write(model_config)

    callbacks = [
        ReduceLROnPlateau(verbose=1, patience=5,
                          factor=0.5, monitor='val_loss'),
        ModelCheckpoint(FLAGS.weights_save,
                        verbose=1, save_weights_only=True),
    ]

    optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)

    if FLAGS.mode == 'resume':
        model.load_weights(FLAGS.weights_save)

    model.compile(optimizer=optimizer, loss=[
                  'binary_crossentropy'], metrics=[ iou_coef, Precision(), Recall()])

    model.summary()

    model.fit(x=Xtrain, y=ytrain, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size,
              validation_data=(Xval, yval), callbacks=callbacks)


if __name__ == '__main__':
    app.run(main)
