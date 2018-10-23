import numpy as np
import tensorflow as tf
from utils import decode

class decoder_CAE():
    def __init__(self,checkpoint_dir):
        self.checkpoint_dir=checkpoint_dir
        tf.reset_default_graph()
        self.img = tf.placeholder(tf.float32, shape=(None,None,None,32),name='x') #PARCHE TEMPORAL
        with tf.variable_scope("decoder"):
            self.decoded=decode(self.img) 

        self.session=tf.Session()
        latest=tf.train.latest_checkpoint(checkpoint_dir=self.checkpoint_dir+'decoder')
        saver=tf.train.Saver()
        saver.restore(self.session, save_path=latest)
        print([v.name for v in tf.global_variables()])

    def uncompress(self,X_in,config=''):
        
        max_val=X_in.max()
        X_decoded=np.round(self.session.run(self.decoded,feed_dict={self.img:X_in/max_val})*max_val).astype(np.uint8)
        print(X_decoded[0].min(),X_decoded[0].max())
        return X_decoded

    def close_session(self):
        self.session.close()
    