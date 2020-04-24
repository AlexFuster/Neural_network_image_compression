from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose
import tensorflow as tf
from utils import convert_to_rgb, convert_to_colourspace, ycbcr_kernel, ycbcr_inv_kernel, ycbcr_off, read_dataset
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt
from encoder import BaseEncoder, Encoder
from decoder import BaseDecoder, Decoder
import os 
from calc_ssim import get_ms_ssim

physical_devices = tf.config.list_physical_devices('GPU') 
try: 
  tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
  # Invalid device or cannot modify virtual devices once initialized. 
  pass 

LOGOF2=tf.math.log(tf.constant(2,dtype=tf.float32))

class Entropy(tf.keras.Model):
    def __init__(self):
        super(Entropy, self).__init__()
        
    def __call__(self,encoded):
        batch_encoded=tf.concat(encoded,axis=0)
        p_=tf.reduce_mean(batch_encoded,axis=[1,2])
        aprox_entropy=tf.expand_dims(tf.reduce_sum(-p_*tf.math.log(p_+1)/LOGOF2,axis=1),-1)
        return tf.split(aprox_entropy, 3, axis=0)

class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.entropy= Entropy()
        self.optimizer_y=tf.keras.optimizers.Adam(1e-4)
        self.optimizer_cbcr=tf.keras.optimizers.Adam(1e-4)
        self.summary_writer = tf.summary.create_file_writer('logs')
        self.train_mode()

    def __call__(self,images):
        img_norm = tf.image.random_flip_left_right(images)
        img_norm = tf.image.random_flip_up_down(img_norm)

        img_channels=self.encoder.preprocess(img_norm)

        res,encoded=self.encoder(img_channels)
        entropy_losses=self.entropy(encoded)
        res=self.decoder(res)

        losses={
            'ssim':[],
            'entropy':[],
            'total':[]    
        }
        for i in range(3):
            ssim_i=tf.image.ssim(img_channels[i], res[i], max_val=1.0)
            losses['ssim'].append(tf.reduce_mean(ssim_i).numpy())
            loss_i=tf.reduce_mean((1-ssim_i)/2 + self.entropy_loss_coef * entropy_losses[i])
            losses['total'].append(loss_i)
            losses['entropy'].append(tf.reduce_mean(entropy_losses[i]).numpy())

        return losses

    def train_step(self,batch):
        with tf.GradientTape() as y_tape, tf.GradientTape() as cbcr_tape:
            losses=self(batch)
            total_losses=losses['total']

            trainable_variables_y=self.encoder.model_y.trainable_variables + self.decoder.model_y.trainable_variables
            trainable_variables_cbcr=self.encoder.model_cbcr.trainable_variables + self.decoder.model_cbcr.trainable_variables

            gradients_y = y_tape.gradient(total_losses[0], trainable_variables_y)
            gradients_cbcr = cbcr_tape.gradient((total_losses[1]+total_losses[2])/2, trainable_variables_cbcr)

        self.optimizer_y.apply_gradients(zip(gradients_y, trainable_variables_y))
        self.optimizer_cbcr.apply_gradients(zip(gradients_cbcr, trainable_variables_cbcr))
        return losses

    def _save(self):
        self.encoder.save_weights('../checkpoints/encoder')
        self.decoder.save_weights('../checkpoints/decoder')

    def train_mode(self):
        self.encoder.train_mode=True
        self.decoder.train_mode=True
    
    def eval_mode(self):
        self.encoder.train_mode=False
        self.decoder.train_mode=False

    def training_loop(self, x, x_val_path, max_epochs, batch_size, entropy_loss_coef):
        train_ds = tf.data.Dataset.from_tensor_slices(x).shuffle(10000).batch(batch_size)
        comp_val_path=x_val_path+'_compressed'
        uncomp_val_path=x_val_path+'_uncompressed'
        step=0
        self.entropy_loss_coef=entropy_loss_coef

        self.encoder.load('../checkpoints/encoder')
        self.decoder.load('../checkpoints/decoder')

        for epoch in range(max_epochs):
            for batch in train_ds:
                losses=self.train_step(batch)
                losses['total']=[aux.numpy() for aux in losses['total']]
                print('EPOCH',epoch,losses)
                if x_val_path is not None and step%100==0:
                    self.eval_mode()
                    self._save()
                    filename=list(self.encoder.compress(x_val_path,'../checkpoints/encoder').keys())[0]
                    size_in_kb=os.path.getsize(os.path.join(comp_val_path,filename+'.png'))/1000
                    out=self.decoder.uncompress(comp_val_path,'../checkpoints/decoder')
                    val_ssim=get_ms_ssim(x_val_path,uncomp_val_path)
                    with self.summary_writer.as_default():
                        tf.summary.scalar('MS-SSIM', val_ssim[filename],step=step)
                        tf.summary.scalar('size(kb)', size_in_kb,step=step)
                        tf.summary.image(filename,tf.expand_dims(list(out.values())[0],0),step=step)
                    self.train_mode()
                step+=1

if __name__ == "__main__":
    imgs,_ = read_dataset('../../data/imagenet_patches')

    training_obj = Autoencoder()
    training_obj.training_loop(imgs, '../../data/kodak_img', 300, 32, 0.01)