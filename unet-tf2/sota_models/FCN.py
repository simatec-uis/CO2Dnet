import os
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, Activation
from utils.BilinearUpSampling import BilinearUpSampling2D, BilinearResize2D





def FCN_Vgg16_16s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=1):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
    # Block 1
    features = 64
    l2_regularizer = tf.keras.regularizers.L2(l2=1e-4)

    x = Conv2D(features, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2_regularizer)(img_input)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2_regularizer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(features*2, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2_regularizer)(x)
    x = Conv2D(features*2, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2_regularizer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(features*4, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2_regularizer)(x)
    x = Conv2D(features*4, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2_regularizer)(x)
    x = Conv2D(features*4, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2_regularizer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(features*6, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2_regularizer)(x)
    x = Conv2D(features*6, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2_regularizer)(x)
    x = Conv2D(features*6, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2_regularizer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(features*6, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2_regularizer)(x)
    x = Conv2D(features*6, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2_regularizer)(x)
    x = Conv2D(features*6, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2_regularizer)(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(int(features*10), (3, 3), activation='relu', padding='same', dilation_rate=(1, 1),
                      name='fc1', kernel_regularizer=l2_regularizer)(x)
    # x = Dropout(0.5)(x)
    x = Conv2D(int(features*10), (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2_regularizer)(x)
    # x = Dropout(0.5)(x)
    #classifying layer
    x = Conv2D(classes, (1, 1), activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2_regularizer)(x)

    # x = BilinearResize2D(size=(320, 320))(x)
    x = UpSampling2D(size=(16, 16), interpolation='bilinear')(x)
    x = Activation('sigmoid')(x)

    model = Model(img_input, x)
    model_name = 'FCN_Vgg16_16'
    return model, model_name