import tensorflow as tf
from tensorflow.keras.layers import *


KERNEL_REGUL = tf.keras.regularizers.l2(1e-8)


class convBlock(tf.keras.layers.Layer):
    def __init__(self, feature, **kwargs):
        super(convBlock, self).__init__(**kwargs)

        self.conv = Conv2D(feature, 3, activation=None, kernel_initializer='he_uniform',    
                            padding="same", kernel_regularizer=KERNEL_REGUL)

        self.bn = BatchNormalization()

        self.relu = ReLU()

    def get_config(self):
        return super().get_config() 

    def call(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)
        return outputs


class inputBlock(tf.keras.layers.Layer):
    def __init__(self, F, **kwargs):
        super(inputBlock, self).__init__(**kwargs)

        self.conv1 = convBlock(F)
        self.conv2 = convBlock(F)

    def get_config(self):
        return super().get_config() 

    def call(self, inputs):

        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)

        return outputs


class downBLock(tf.keras.layers.Layer):
    def __init__(self, F, lvl, **kwargs):
        super(downBLock, self).__init__(**kwargs)

        feature = F*(2**lvl)
        self.down = MaxPooling2D(pool_size=(2, 2))
        self.conv1 = convBlock(feature)
        self.conv2 = convBlock(feature)

    def get_config(self):
        return super().get_config()                     

    def call(self, inputs):

        outputs = self.down(inputs)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)

        return outputs


class blottleBlock(tf.keras.layers.Layer):
    def __init__(self, F, lvl, **kwargs):
        super(blottleBlock, self).__init__(**kwargs)

        feature = F*(2**lvl)
        self.down = MaxPooling2D(pool_size=(2, 2))
        self.conv1 = convBlock(feature)
        self.conv2 = convBlock(feature)

    def get_config(self):
        return super().get_config() 

    def call(self, inputs):

        outputs = self.down(inputs)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)

        return outputs

    
class upBlock(tf.keras.layers.Layer):
    def __init__(self, F, lvl, **kwargs):
        super(upBlock, self).__init__(**kwargs)

        feature = F*(2**lvl)
        self.up = UpSampling2D(size=(2, 2))
        # self.up = Conv2DTranspose(feature, 2, strides=(2,2) , padding='same')
        self.concat = Concatenate()
        self.conv1 = convBlock(feature)
        self.conv2 = convBlock(feature)
    def get_config(self):
        return super().get_config() 

    def call(self, inputs):

        outputs = self.up(inputs[0])
        outputs = self.concat([outputs, inputs[1]])
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)

        return outputs

class outBlock(tf.keras.layers.Layer):
    def __init__(self, F, **kwargs):
        super(outBlock, self).__init__(**kwargs)

        # self.up = Conv2DTranspose(F, 2, strides=(2,2) , padding='same')
        self.up = UpSampling2D(size=(2, 2))
        self.concat = Concatenate()
        self.conv1 = convBlock(F)
        self.conv2 = convBlock(F)

    def get_config(self):
        return super().get_config() 

    def call(self, inputs):

        outputs = self.up(inputs[0])
        outputs = self.concat([outputs, inputs[1]])
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)

        return outputs
