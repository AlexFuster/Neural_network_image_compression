import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose
from utils import convert_to_rgb, convert_to_colourspace, ycbcr_kernel, ycbcr_inv_kernel, ycbcr_off, read_dataset
import numpy as np
from PIL import Image
from encoder import BaseEncoder, Encoder
from decoder import BaseDecoder, Decoder
import os
#import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

get_png_size = lambda xin: tf.strings.length(tf.image.encode_png(xin))

def get_bpp(encoded,tot_pixels_compressed=None):
    img_to_convert=tf.cast(tf.round(encoded), tf.uint8)
    #transform_shape = tf.cast(tf.cast(tf.shape(encoded), tf.float32) *tf.convert_to_tensor([1, 4, 8, 1 / 32.0]), tf.int32)
    #img_to_convert = tf.reshape(tf.cast(tf.round(encoded), tf.uint8),transform_shape)
    #if tot_pixels_compressed is None:
    #    tot_pixels_compressed=tf.cast(tf.reduce_prod(tf.shape(img_to_convert)[1:3]),tf.float32)
    size_list = tf.cast(tf.stack(tf.map_fn(get_png_size,img_to_convert,dtype=tf.int32)), tf.float32)
    bpp = 8 * tf.reshape(size_list, (-1, 1)) / tot_pixels_compressed
    return bpp

_MODELS = ['Y','CbCr']

class Training:
    def __init__(self):

        self.encoder_models = [BaseEncoder(),BaseEncoder()]
        self.decoder_models = [BaseDecoder(),BaseDecoder()]
        self.epoch = 0

        #self.entropy_model = Entropynet()
        self.summary_writer = tf.summary.create_file_writer('logs')

    def __call__(self, x, x_val_path, max_epochs, batch_size, entropy_loss_coef):
        logof2=tf.math.log(tf.constant(2,dtype=tf.float32))

        optimizer_y = tf.keras.optimizers.Adam(1e-4)
        optimizer_cbcr = tf.keras.optimizers.Adam(1e-4)
        tot_pixels_compressed=x.shape[1] * x.shape[2]
        train_ds = tf.data.Dataset.from_tensor_slices(x).shuffle(10000).batch(batch_size)
        step=0
        comp_val_path=x_val_path+'_compressed'
        encoder=Encoder()
        decoder=Decoder()
        img_val,filenames_val=read_dataset(x_val_path)
        img_val_shape_dict={}
        for i in range(len(filenames_val)):
            curr_img_shape=img_val[i].shape
            img_val_shape_dict[filenames_val[i]]=curr_img_shape[-2]*curr_img_shape[-3]
        img_val=None

        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch
            for images in train_ds:
                with tf.GradientTape() as y_tape, tf.GradientTape() as cbcr_tape, tf.GradientTape():
                    img_norm = images / 255
                    img_norm = tf.image.random_flip_left_right(img_norm)
                    img_norm = tf.image.random_flip_up_down(img_norm)
                    img_channels = convert_to_colourspace(ycbcr_kernel, ycbcr_off, img_norm)

                    img_channels_0=img_channels[0]
                    img_channels_1=tf.concat(img_channels[1:],axis=0)

                    encoded_0=self.encoder_models[0](img_channels_0)
                    encoded_1=self.encoder_models[1](img_channels_1)

                    #noisy_encoded_0=tf.clip_by_value(encoded_0 + tf.random.uniform(tf.shape(encoded_0), -0.5, 0.5) / 255, 0, 1)
                    #noisy_encoded_1=tf.clip_by_value(encoded_1 + tf.random.uniform(tf.shape(encoded_1), -0.5, 0.5) / 255, 0, 1)

                    noisy_encoded_0=tf.stack([encoded_0[:,:,:,i]*i for i in range(256)],axis=-1)
                    noisy_encoded_1=tf.stack([encoded_1[:,:,:,i]*i for i in range(256)],axis=-1)

                    noisy_encoded_0=tf.reduce_sum(noisy_encoded_0,axis=-1,keepdims=True)/255
                    noisy_encoded_1=tf.reduce_sum(noisy_encoded_1,axis=-1,keepdims=True)/255

                    batch_encoded = tf.concat([encoded_0,encoded_1], axis=0)
                    p_=tf.reduce_mean(batch_encoded,axis=[1,2])
                    aprox_entropy=tf.reshape(tf.reduce_sum(-p_*tf.math.log(p_+1)/logof2,axis=1),(-1,1))
                    #aprox_entropy = self.entropy_model(batch_encoded)

                    encoded_denorm = batch_encoded * 255
                    encoded_denorm=tf.math.argmax(encoded_denorm,axis=3)
                    encoded_denorm=tf.expand_dims(encoded_denorm,axis=-1)

                    #bpp=get_bpp(encoded_denorm,tot_pixels_compressed)
                    #true_bpp=get_bpp(tf.concat(tf.split(img_to_convert,3,axis=0),axis=3),tot_pixels_compressed)

                    #true_bpp=true_bpp.numpy().mean()

                    #aprox_entropy_loss = tf.reduce_mean(
                    #    tf.keras.losses.MSE(bpp, aprox_entropy))

                    entropy_losses = tf.split(aprox_entropy, 3, axis=0)

                    #bpp_channels = tf.split(bpp, 3, axis=0)

                    decoded_0 = self.decoder_models[0](noisy_encoded_0)
                    ssim_0=tf.image.ssim(img_channels_0, decoded_0, max_val=1.0)
                    loss_0=(1 - ssim_0) / 2 + entropy_loss_coef * entropy_losses[0]
                    #loss_0=tf.reduce_mean((img_channels_0-decoded_0)**2) + entropy_loss_coef * entropy_losses[0]

                    decoded_1 = self.decoder_models[1](noisy_encoded_1)
                    ssim_1=tf.image.ssim(img_channels_1, decoded_1, max_val=1.0)
                    ssim_cb,ssim_cr=tf.split(ssim_1,2,axis=0)
                    ssim_cb=tf.reduce_mean(ssim_cb)
                    ssim_cr=tf.reduce_mean(ssim_cr)

                    loss_1=(1 - ssim_1) / 2 + entropy_loss_coef *0.001* tf.concat(entropy_losses[1:],axis=0)
                    #loss_1=tf.reduce_mean((img_channels_1-decoded_1)**2) + entropy_loss_coef * tf.concat(entropy_losses[1:],axis=0)

                    ssims=[ssim_0.numpy().mean(),ssim_cb.numpy(),ssim_cr.numpy()]
                    #bpp_res=[aux_bpp.numpy().mean() for aux_bpp in bpp_channels]
                    decoded = [decoded_0]+tf.split(decoded_1,2,axis=0)

                #with self.summary_writer.as_default():
                #    tf.summary.scalar('SSIM_Y', ssim_0,step=step)
                #    tf.summary.scalar('SSIM_Cb', ssim_cb,step=step)
                #    tf.summary.scalar('SSIM_Cr', ssim_cr,step=step)

                print('EPOCH:', epoch, 'SSIM:', ssims, 'entropy coef:', entropy_loss_coef, 'entropy:', [aux_ent.numpy().mean() for aux_ent in entropy_losses])
                #print('EPOCH:', epoch, 'SSIM:', ssims)
                main_variables_y = self.encoder_models[
                    0].trainable_variables + self.decoder_models[
                        0].trainable_variables
                main_variables_cbcr = self.encoder_models[
                    1].trainable_variables + self.decoder_models[
                        1].trainable_variables


                gradients_y = y_tape.gradient(loss_0, main_variables_y)
                gradients_cbcr = cbcr_tape.gradient(loss_1, main_variables_cbcr)


                optimizer_y.apply_gradients(zip(gradients_y, main_variables_y))
                optimizer_cbcr.apply_gradients(zip(gradients_cbcr, main_variables_cbcr))

                step+=1
                if x_val_path is not None and step%10==0:

                    #decoded = tf.clip_by_value(
                    #     convert_to_rgb(ycbcr_inv_kernel, ycbcr_off, *decoded), 0, 1)
                    #aux_enc=(tf.concat([noisy_encoded_0]+tf.split(noisy_encoded_1,2,axis=0),axis=-1).numpy()*255).round().astype(np.uint8)[0,:,:,:]
                    #Image.fromarray(aux_enc).save('kodakim0_c.png')
                    #decoded=(decoded.numpy()*255).round().astype(np.uint8)[0,:,:,:]
                    #Image.fromarray(decoded).save('kodakim0_u.png')

                    self._save()
                    encoder.compress(x_val_path,'../checkpoints/encoder')
                    decoder.uncompress(comp_val_path,'../checkpoints/decoder')

                    bpp_filepath=comp_val_path+'/val_bpp.txt'
                    try:
                        os.remove(bpp_filepath)
                    except:
                        pass
                    with open(bpp_filepath,'a+') as f:
                        for filename in os.listdir(comp_val_path):
                            if filename.endswith('.png'):
                                bpp_val=8*os.path.getsize(comp_val_path+'/'+filename)/img_val_shape_dict[filename.replace('.png','')]
                                f.write(filename+'\t'+str(bpp_val)+'\n')

            entropy_loss_coef= min(entropy_loss_coef*2,0.1)

    def _save(self):
        for i,name in enumerate(_MODELS):
            self.encoder_models[i].save_weights('../checkpoints/encoder{}'.format(name))
            self.decoder_models[i].save_weights('../checkpoints/decoder{}'.format(name))

        print('checkpoint saved')


if __name__ == "__main__":
    imgs,_ = read_dataset('../../data/imagenet_patches_256')

    training_obj = Training()
    training_obj(imgs, '../../data/kodak_img', 30, 16, 0.001)