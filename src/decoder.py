from utils import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose
from utils import convert_to_rgb, ycbcr_inv_kernel, ycbcr_off

class BaseDecoder(tf.keras.Model):
    def __init__(self):
        super(BaseDecoder, self).__init__()
        self.dconv1 = Conv2DTranspose(64,5,2,'SAME',activation=tf.nn.leaky_relu)
        #self.dconv2 = Conv2DTranspose(64,3,1,'SAME',activation=tf.nn.leaky_relu)
        #self.dconv3 = Conv2DTranspose(64,3,1,'SAME',activation=tf.nn.leaky_relu)
        #self.dconv4 = Conv2DTranspose(64,5,2,'SAME',activation=tf.nn.leaky_relu)
        self.dconv5 = Conv2DTranspose(64,3,1,'SAME',activation=tf.nn.leaky_relu)
        self.dconv6 = Conv2DTranspose(64,3,1,'SAME',activation=tf.nn.leaky_relu)
        self.dconv7 = Conv2DTranspose(32,5,1,'SAME',activation=tf.nn.leaky_relu)
        self.dconv8 = Conv2DTranspose(1,5,2,'SAME',activation=tf.nn.leaky_relu)

    def call(self, x):
        x = self.dconv1(x)
        #res = x
        #x = self.dconv2(x)
        #x = self.dconv3(x)
        #x = x + res
        #x = self.dconv4(x)
        res = x
        x = self.dconv5(x)
        x = self.dconv6(x)
        x = x + res
        x = self.dconv7(x)
        x = self.dconv8(x)
        return tf.clip_by_value(x, 0, 1)


class Decoder(Model):
    def __init__(self):
        super(Decoder, self).__init__(BaseDecoder)

    def preprocess(self,x):
        x = tf.cast(x,tf.float32) / 255
        return x

    def postprocess(self,x):
        return tf.clip_by_value(
            convert_to_rgb(ycbcr_inv_kernel, ycbcr_off, *x), 0, 1)*255

    def __call__(self, x):
        if not self.train_mode:
            x=self.preprocess(x)

        x = tf.split(x, 3, axis=3)
        decoded = self.run_model(x)

        if not self.train_mode:
            decoded=self.postprocess(decoded)

        return decoded

    def uncompress(self,dataset_path,checkpoint_path):
        output_dir=dataset_path.replace('compressed','uncompressed')
        return self._use_model(dataset_path,checkpoint_path,output_dir,in_cshape=96)