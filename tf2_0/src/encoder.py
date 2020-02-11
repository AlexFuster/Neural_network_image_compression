from utils import ProClass
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

class Encoder(ProClass):
    def __init__(self):
        super(Encoder, self).__init__(BaseEncoder)

    def __call__(self, x):
        img_norm = x.astype(np.float32) / 255

        img_channels = convert_to_colourspace(ycbcr_kernel, ycbcr_off, img_norm)

        encoded = self.run_model(img_channels)

        aux=[]
        for i in range(3):
            aux.append(np.stack([encoded[i][:,:,:,j]*j for j in range(256)],axis=-1).sum(axis=-1))

        encoded=np.stack(aux,axis=-1).round().astype(np.uint8)
        #encoded=np.stack([tf.math.argmax(encoded[i],axis=-1) for i in range(3)],axis=-1).astype(np.uint8)

        return encoded

    def compress(self,dataset_path,checkpoint_path):
        output_dir=dataset_path+'_compressed'
        self._use_model(dataset_path,checkpoint_path,output_dir,in_cshape=3)