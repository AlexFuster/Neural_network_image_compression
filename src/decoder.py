import numpy as np
import tensorflow as tf
from utils import decode
from utils import convert_to_rgb, convert_to_colourspace,ycbcr_inv_kernel,ycbcr_off,pca_inv_kernel,pca_off
import matplotlib.pyplot as plt
def reshape_to_tensor(tensor):
    tensor_shape=tf.shape(tensor)
    return tf.reshape(tensor,(tensor_shape[0],tensor_shape[1]//4,tensor_shape[2]//8,32))

class decoder_CAE():
    def __init__(self,checkpoint_dir):
        self.checkpoint_dir=checkpoint_dir
        tf.reset_default_graph()
        self.img = tf.placeholder(tf.float32, shape=(None,None,None,3),name='x') #PARCHE TEMPORAL
        img_1,img_2,img_3=tf.split(self.img,3,axis=3)
        
        img_1=reshape_to_tensor(img_1)
        img_2=reshape_to_tensor(img_2)
        img_3=reshape_to_tensor(img_3)

        with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE):
            decoded_1=tf.clip_by_value(decode(img_1,0),0,1)
            decoded_2=tf.clip_by_value(decode(img_2,1),0,1)
            decoded_3=tf.clip_by_value(decode(img_3,1),0,1)

        self.decoded=tf.clip_by_value(convert_to_rgb(ycbcr_inv_kernel,ycbcr_off,decoded_1,decoded_2,decoded_3),0,1)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session=tf.Session(config=config)
        latest=tf.train.latest_checkpoint(checkpoint_dir=self.checkpoint_dir+'decoder')
        saver=tf.train.Saver()
        saver.restore(self.session, save_path=latest)
        print([v.name for v in tf.global_variables()])

    def uncompress(self,X_in):
        
        if X_in.dtype==np.uint8:
            max_val=255
        assert max_val==255

        X_decoded=np.round(self.session.run(self.decoded,feed_dict={self.img:X_in/max_val})*max_val).astype(np.uint8)
        print(X_decoded.min(),X_decoded.max())

        return X_decoded

    def close_session(self):
        self.session.close()
    