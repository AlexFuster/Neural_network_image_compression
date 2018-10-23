import numpy as np
import tensorflow as tf
from utils import encode, code_array

class encoder_CAE():
    def __init__(self,checkpoint_dir):
        self.checkpoint_dir=checkpoint_dir
        tf.reset_default_graph()
        self.img = tf.placeholder(tf.float32, shape=(None,None,None,1),name='x')
        with tf.variable_scope("encoder"):
            self.encoded=encode(self.img)

        self.session=tf.Session()
        latest=tf.train.latest_checkpoint(checkpoint_dir=self.checkpoint_dir+'encoder')
        saver=tf.train.Saver()
        saver.restore(self.session, save_path=latest)
        print([v.name for v in tf.global_variables()])

    def compress(self,X_in):
        max_val=X_in.max()

        X_encoded=np.round(self.session.run(self.encoded,feed_dict={self.img:X_in/max_val})*max_val)
        return code_array(X_encoded,max_val)

    def close_session(self):
        self.session.close()