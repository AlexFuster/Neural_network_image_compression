from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose
import tensorflow as tf
from utils import convert_to_rgb, convert_to_colourspace, ycbcr_kernel, ycbcr_inv_kernel, ycbcr_off, read_dataset
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt

get_png_size = lambda xin: tf.strings.length(tf.image.encode_png(xin))

def get_bpp(encoded,tot_pixels_compressed=None):
    transform_shape = tf.cast(tf.cast(tf.shape(encoded), tf.float32) *tf.convert_to_tensor([1, 4, 8, 1 / 32.0]), tf.int32)
    img_to_convert = tf.reshape(tf.cast(tf.round(encoded), tf.uint8),transform_shape)
    if tot_pixels_compressed is None:
        tot_pixels_compressed=tf.cast(tf.reduce_prod(tf.shape(img_to_convert)[1:3]),tf.float32)
    size_list = tf.cast(tf.stack(tf.map_fn(get_png_size,img_to_convert,dtype=tf.int32)), tf.float32)
    bpp = 8 * tf.reshape(size_list, (-1, 1)) / tot_pixels_compressed
    return bpp

_MODELS = ['Y','CbCr']

class BaseEncoder(tf.keras.Model):
    def __init__(self):
        super(BaseEncoder, self).__init__()
        self.conv1 = Conv2D(32, 5, 2, 'SAME', activation=tf.nn.leaky_relu)
        self.conv2 = Conv2D(64, 5, 2, 'SAME', activation=tf.nn.leaky_relu)
        self.conv3 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        self.conv4 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        #self.conv5 = Conv2D(64, 5, 2, 'SAME', activation=tf.nn.leaky_relu)
        #self.conv6 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        #self.conv7 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        self.conv8 = Conv2D(32, 5, 2, 'SAME', activation=tf.nn.leaky_relu)

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


class BaseDecoder(tf.keras.Model):
    def __init__(self):
        super(BaseDecoder, self).__init__()
        self.dconv1 = Conv2DTranspose(64,5,2,'SAME',activation=tf.nn.leaky_relu)
        #self.dconv2 = Conv2DTranspose(64,3,1,'SAME',activation=tf.nn.leaky_relu)
        #self.dconv3 = Conv2DTranspose(64,3,1,'SAME',activation=tf.nn.leaky_relu)
        #self.dconv4 = Conv2DTranspose(64,5,2,'SAME',activation=tf.nn.leaky_relu)
        self.dconv5 = Conv2DTranspose(64,3,1,'SAME',activation=tf.nn.leaky_relu)
        self.dconv6 = Conv2DTranspose(64,3,1,'SAME',activation=tf.nn.leaky_relu)
        self.dconv7 = Conv2DTranspose(64,5,2,'SAME',activation=tf.nn.leaky_relu)
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


class Entropynet(tf.keras.Model):
    def __init__(self):
        super(Entropynet, self).__init__()
        self.conv1 = Conv2D(64, 5, 2, 'SAME', activation=tf.nn.leaky_relu)
        self.conv2 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        self.conv3 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        self.flatten = Flatten()
        self.dense1 = Dense(512)
        self.dense2 = Dense(1)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return tf.clip_by_value(x, 0, 8)


class ProClass:
    def __init__(self,model_class):
        self.models=[model_class(),model_class()]

    def run_model(self,x):
        return [
            self.models[0](x[0]),
            self.models[1](x[1]),
            self.models[1](x[2])
        ]

    def load(self,path):
        for i,name in enumerate(_MODELS):
            self.models[i].load_weights(path+str(name))


class Encoder(ProClass):
    def __init__(self):
        super(Encoder, self).__init__(BaseEncoder)

    def __call__(self, x):
        img_norm = x.astype(np.float32) / 255

        img_channels = convert_to_colourspace(ycbcr_kernel, ycbcr_off, img_norm)

        encoded = self.run_model(img_channels)

        encoded = tf.concat(encoded, axis=3)

        return np.round(encoded * 255)


class Decoder(ProClass):
    def __init__(self):
        super(Decoder, self).__init__(BaseDecoder)

    def __call__(self, x):
        img_norm = x.astype(np.float32) / 255
        img_channels = tf.split(img_norm, 3, axis=3)

        decoded = self.run_model(img_channels)

        decoded = tf.clip_by_value(
            convert_to_rgb(ycbcr_inv_kernel, ycbcr_off, *decoded), 0, 1)

        return np.round(decoded * 255).astype(np.uint8)


class Training:
    def __init__(self):

        self.encoder_models = [BaseEncoder(),BaseEncoder()]
        self.decoder_models = [BaseDecoder(),BaseDecoder()]
        self.epoch = 0

        self.entropy_model = Entropynet()
        self.summary_writer = tf.summary.create_file_writer('logs')

    def __call__(self, x, x_val, max_epochs, batch_size, entropy_loss_coef):

        optimizer_y = tf.keras.optimizers.Adam(1e-4)
        optimizer_cbcr = tf.keras.optimizers.Adam(1e-4)
        optimizer_entropy = tf.keras.optimizers.Adam(1e-4)
        tot_pixels_compressed=x.shape[1] * x.shape[2]
        train_ds = tf.data.Dataset.from_tensor_slices(x).shuffle(10000).batch(batch_size)
        step=0
        encoder=Encoder()
        decoder=Decoder()

        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch
            for images in train_ds:
                with tf.GradientTape() as y_tape, tf.GradientTape() as cbcr_tape, tf.GradientTape() as entropy_tape:
                    img_norm = images / 255
                    img_norm = tf.image.random_flip_left_right(img_norm)
                    img_norm = tf.image.random_flip_up_down(img_norm)
                    img_channels = convert_to_colourspace(ycbcr_kernel, ycbcr_off, img_norm)

                    img_channels_0=img_channels[0]
                    img_channels_1=tf.concat(img_channels[1:],axis=0)

                    encoded_0=self.encoder_models[0](img_channels_0)
                    encoded_1=self.encoder_models[1](img_channels_1)

                    noisy_encoded_0=tf.clip_by_value(encoded_0 + tf.random.uniform(tf.shape(encoded_0), -0.5, 0.5) / 255, 0, 1)
                    noisy_encoded_1=tf.clip_by_value(encoded_1 + tf.random.uniform(tf.shape(encoded_1), -0.5, 0.5) / 255, 0, 1)

                    batch_encoded = tf.concat([encoded_0,encoded_1], axis=0)
                    aprox_entropy = self.entropy_model(batch_encoded)

                    encoded_denorm = batch_encoded * 255

                    bpp=get_bpp(encoded_denorm,tot_pixels_compressed)
                    #true_bpp=get_bpp(tf.concat(tf.split(img_to_convert,3,axis=0),axis=3),tot_pixels_compressed)

                    #true_bpp=true_bpp.numpy().mean()

                    aprox_entropy_loss = tf.reduce_mean(
                        tf.keras.losses.MSE(bpp, aprox_entropy))

                    entropy_losses = tf.split(aprox_entropy, 3, axis=0)

                    bpp_channels = tf.split(bpp, 3, axis=0)

                    decoded_0 = self.decoder_models[0](noisy_encoded_0)
                    ssim_0=tf.reduce_mean(tf.image.ssim(img_channels_0, decoded_0, max_val=1.0))
                    loss_0=(1 - ssim_0) / 2 + entropy_loss_coef * entropy_losses[0]
                    #loss_0=tf.reduce_mean((img_channels_0-decoded_0)**2) + entropy_loss_coef * entropy_losses[0]

                    decoded_1 = self.decoder_models[1](noisy_encoded_1)
                    ssim_1=tf.image.ssim(img_channels_1, decoded_1, max_val=1.0)
                    ssim_cb,ssim_cr=tf.split(ssim_1,2,axis=0)
                    ssim_cb=tf.reduce_mean(ssim_cb)
                    ssim_cr=tf.reduce_mean(ssim_cr)
                    ssim_1=tf.reduce_mean(ssim_1)

                    loss_1=(1 - ssim_1) / 2 + entropy_loss_coef/2 * tf.concat(entropy_losses[1:],axis=0)
                    #loss_1=tf.reduce_mean((img_channels_1-decoded_1)**2) + entropy_loss_coef * tf.concat(entropy_losses[1:],axis=0)

                    ssims=[ssim_0.numpy(),ssim_cb.numpy(),ssim_cr.numpy()]
                    bpp_res=[aux_bpp.numpy().mean() for aux_bpp in bpp_channels]
                    decoded = [decoded_0]+tf.split(decoded_1,2,axis=0)

                #with self.summary_writer.as_default():
                #    tf.summary.scalar('SSIM_Y', ssim_0,step=step)
                #    tf.summary.scalar('SSIM_Cb', ssim_cb,step=step)
                #    tf.summary.scalar('SSIM_Cr', ssim_cr,step=step)

                print('EPOCH:', epoch, 'SSIM:', ssims, 'BPP:', bpp_res,'Entropy loss:', aprox_entropy_loss.numpy())
                #print('EPOCH:', epoch, 'SSIM:', ssims)
                main_variables_y = self.encoder_models[
                    0].trainable_variables + self.decoder_models[
                        0].trainable_variables
                main_variables_cbcr = self.encoder_models[
                    1].trainable_variables + self.decoder_models[
                        1].trainable_variables

                entropy_variables = self.entropy_model.trainable_variables

                gradients_y = y_tape.gradient(loss_0, main_variables_y)
                gradients_cbcr = cbcr_tape.gradient(loss_1, main_variables_cbcr)
                gradients_entropy = entropy_tape.gradient(aprox_entropy_loss, entropy_variables)


                optimizer_y.apply_gradients(zip(gradients_y, main_variables_y))
                optimizer_cbcr.apply_gradients(zip(gradients_cbcr, main_variables_cbcr))
                optimizer_entropy.apply_gradients(zip(gradients_entropy, entropy_variables))

                step+=1
                if x_val is not None and step%10==0:
                    self._save()

                    encoder.load('checkpoints/encoder')
                    decoder.load('checkpoints/decoder')
                    encoded_val=encoder(x_val)
                    decoded_val=decoder(encoded_val)
                    encoded_val=tf.convert_to_tensor(encoded_val)
                    bpp_val=get_bpp(encoded_val,x_val.shape[1]*x_val.shape[2]).numpy()[0,0]
                    comparison_image=np.concatenate([x_val[0],decoded_val[0]],axis=1).astype(np.uint8)
                    Image.fromarray(comparison_image).save('prueba.png')
                    with open('prueba.txt','w') as f:
                        f.write(str(bpp_val))

            entropy_loss_coef+=0.01

    def _save(self):
        for i,name in enumerate(_MODELS):
            self.encoder_models[i].save_weights('checkpoints/encoder{}'.format(name))
            self.decoder_models[i].save_weights('checkpoints/decoder{}'.format(name))

        print('checkpoint saved')


if __name__ == "__main__":
    imgs,_ = read_dataset('../data/imagenet_patches')
    img_val,_=read_dataset('../data/kodak_img')

    training_obj = Training()
    training_obj(imgs, img_val, 15, 64, 0.01)

    encoder=Encoder()
    encoder.load('checkpoints/encoder')
    decoder=Decoder()
    decoder.load('checkpoints/decoder')

    enc_out=encoder(imgs[:10])
    print(enc_out.shape)
    dec_out=decoder(enc_out)
    print(dec_out.shape)
    print(imgs.max(),dec_out.max())