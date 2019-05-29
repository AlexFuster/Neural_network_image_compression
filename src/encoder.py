import numpy as np
import tensorflow as tf
from utils import encode
from utils import convert_to_rgb, convert_to_colourspace,ycbcr_kernel,ycbcr_off,pca_kernel,pca_off


def reshape_to_image(tensor):
    tensor_shape=tf.shape(tensor)
    return tf.reshape(tensor,(tensor_shape[0],tensor_shape[1]*4,tensor_shape[2]*8,1))

class encoder_CAE():
    def __init__(self,checkpoint_dir):
        self.checkpoint_dir=checkpoint_dir
        tf.reset_default_graph()
        self.img = tf.placeholder(tf.float32, shape=(None,None,None,3),name='x')
        img_norm=self.img
        img_1,img_2,img_3=convert_to_colourspace(ycbcr_kernel,ycbcr_off,img_norm)
        with tf.variable_scope("encoder",reuse=tf.AUTO_REUSE):
            encoded_1=tf.clip_by_value(encode(img_1,0),0,1)
            encoded_2=tf.clip_by_value(encode(img_2,1),0,1)
            encoded_3=tf.clip_by_value(encode(img_3,1),0,1)

        encoded=tf.concat([reshape_to_image(encoded_1),reshape_to_image(encoded_2),reshape_to_image(encoded_3)],axis=3)
        self.encoded=tf.clip_by_value(encoded,0,1)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session=tf.Session(config=config)
        latest=tf.train.latest_checkpoint(checkpoint_dir=self.checkpoint_dir+'encoder')
        saver=tf.train.Saver()
        saver.restore(self.session, save_path=latest)
        print([v.name for v in tf.global_variables()])

    def compress(self,X_in):
        if X_in.dtype==np.uint8:
            max_val=255
        assert max_val==255
        
        X_encoded=np.round(self.session.run(self.encoded,feed_dict={self.img:X_in/max_val})*max_val)
        print(X_encoded.min(),X_encoded.max())
        return X_encoded.astype(np.uint8)

    def close_session(self):
        self.session.close()