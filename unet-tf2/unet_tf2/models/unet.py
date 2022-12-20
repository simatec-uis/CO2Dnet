import tensorflow as tf
from tensorflow.python.keras.layers.core import Lambda
from unet_tf2.models.layers import inputBlock, downBLock, blottleBlock, upBlock, outBlock
from tensorflow.keras.models import *
from tensorflow.keras.layers import Input, Conv2D, Reshape

class UnetLayer(tf.keras.layers.Layer):
    def __init__(self, feature, levels, **kwargs):
        
        self.feature = feature
        self.levels = levels

        super().__init__(**kwargs)

        self.inputBlock = inputBlock(feature)
        self.downBlocks = [ downBLock(feature,level) for level in range(1,levels)  ]
        self.blottleBlock = blottleBlock(feature, levels)
        self.upBlocks = [ upBlock(feature,levels - level) for level in range(1,levels)]
        self.outBlock = outBlock(feature)

    def get_config(self):
        config = super().get_config()    
        config['feature'] = self.feature
        config['levels'] = self.levels
        return config

    def call(self, inputs):

        outputs = []
        x = self.inputBlock(inputs)
        outputs.append(x)

        for downlayer in self.downBlocks:
            x = downlayer(x)
            outputs.append(x)

        x = self.blottleBlock(x)    

        for uplayer in self.upBlocks:
            y = outputs.pop(-1)
            x = uplayer([x,y])

        y = outputs.pop(-1)
        x = self.outBlock([x,y])

        return x

def Unet(input_shape, feature=8, levels=3, num_classes=3):

    _input = Input(input_shape)

    unet = UnetLayer(feature, levels)(_input)    

    _output = Conv2D(1, 1, activation='sigmoid', padding="same")(unet)
    # _output = Lambda( lambda x:  tf.argmax( x , -1) )(_output)

    model = Model(_input, _output, name="Unet")
    return model