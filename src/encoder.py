from utils import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose
from utils import convert_to_colourspace, ycbcr_kernel, ycbcr_off

class BaseEncoder(tf.keras.Model):
    def __init__(self):
        super(BaseEncoder, self).__init__()
        self.conv1 = Conv2D(32, 5, 2, 'SAME', activation=tf.nn.leaky_relu)
        self.conv2 = Conv2D(64, 5, 1, 'SAME', activation=tf.nn.leaky_relu)
        self.conv3 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        self.conv4 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        #self.conv5 = Conv2D(64, 5, 2, 'SAME', activation=tf.nn.leaky_relu)
        #self.conv6 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        #self.conv7 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        #self.conv8 = Conv2D(32, 5, 2, 'SAME', activation=tf.nn.leaky_relu)
        self.conv8 = Conv2D(256, 5, 2, 'SAME', activation=tf.nn.softmax)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        res = x
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + res
        #x = self.conv5(x)
        #res = x
        #x = self.conv6(x)
        #x = self.conv7(x)
        #x = x + res
        x = self.conv8(x)
        return tf.clip_by_value(x, 0, 1)

class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__(BaseEncoder)

    def preprocess(self,x):
        x = tf.cast(x,tf.float32) / 255
        return convert_to_colourspace(ycbcr_kernel, ycbcr_off, x)

    def __call__(self, x):
        if not self.train_mode:
            x=self.preprocess(x)
        encoded = self.run_model(x)
        
        tf_range=tf.cast(tf.range(256),tf.float32)
        res=tf.stack([tf.reduce_sum(encoded[i]*tf_range,-1) for i in range(3)],-1)
        if self.train_mode:
            return res/255,encoded
        else:
            return res

    def compress(self,dataset_path,checkpoint_path):
        output_dir=dataset_path+'_compressed'
        return self._use_model(dataset_path,checkpoint_path,output_dir,in_cshape=3)