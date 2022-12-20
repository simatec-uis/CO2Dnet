from absl import app, flags, logging
from absl.flags import FLAGS

import os
import json
import numpy as np
import cv2

from unet_tf2.utils import iou_coef
from unet_tf2.models.unet import UnetLayer
from tensorflow import keras


flags.DEFINE_string('image', '', 'path to dataset')
flags.DEFINE_string('weights', './checkpoints/unet_train.tf', 'path to save file')
flags.DEFINE_string('output', './segmentation.jpg', 'path to save file')
# flags.DEFINE_integer('size', 1024, 'levels of Unet network')

SIZE = 640

def main(_argv):
    img = np.expand_dims(np.asarray(cv2.imread(FLAGS.image)),0)
    M , N , C = img.shape[1:]

    down_img = cv2.resize( img[0] , (SIZE,SIZE), cv2.INTER_AREA)
    down_img = np.expand_dims(down_img, 0)

    dependencies = {
        'UnetLayer': UnetLayer,
        'iou_coef' : iou_coef,
    }
    
    model_name , _ = os.path.splitext(FLAGS.weights)
    config_path = model_name + ".txt"

    with open(config_path) as file:
        config = json.load(file)

    config_string = json.dumps(config)

    model = keras.models.model_from_json(config_string, custom_objects=dependencies)
    model.load_weights(FLAGS.weights)

    segm = model.predict(down_img/255) 
    segm = cv2.resize(segm[0], (N, M), cv2.INTER_AREA)
    segm = np.expand_dims(segm, 0)
    segm = np.expand_dims(segm, -1)
    segm = (segm > 0.5)*0.5 + 0.5
    segm = img*segm 
    segm = segm[0]
    segm = segm.astype('uint8')
    
    print('SHAPE: ',segm.shape)
    print('TYPE: ', type(segm))
    
    img = cv2.cvtColor(segm, cv2.COLOR_RGB2BGR)
    cv2.imwrite(FLAGS.output, img)
    logging.info('segmentation saved to: {}'.format(FLAGS.output))

if __name__ == '__main__':
    app.run(main)