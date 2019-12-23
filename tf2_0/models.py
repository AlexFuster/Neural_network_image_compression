from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose
import tensorflow as tf
from utils import convert_to_rgb, convert_to_colourspace, ycbcr_kernel, ycbcr_inv_kernel, ycbcr_off, read_dataset
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt

get_png_size = lambda xin: tf.strings.length(tf.image.encode_png(xin))
_CHANNELS = 3

class BaseEncoder(tf.keras.Model):
    def __init__(self):
        super(BaseEncoder, self).__init__()
        #self.conv1 = Conv2D(32, 5, 2, 'SAME', activation=tf.nn.leaky_relu)
        self.conv2 = Conv2D(64, 5, 2, 'SAME', activation=tf.nn.leaky_relu)
        #self.conv3 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        #self.conv4 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        #self.conv5 = Conv2D(64, 5, 2, 'SAME', activation=tf.nn.leaky_relu)
        #self.conv6 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        #self.conv7 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        self.conv8 = Conv2D(32, 5, 2, 'SAME', activation=tf.nn.leaky_relu)

    def call(self, x):
        #x = self.conv1(x)
        x = self.conv2(x)
        #res = x
        #x = self.conv3(x)
        #x = self.conv4(x)
        #x = x + res
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
        #self.dconv1 = Conv2DTranspose(64,5,2,'SAME',activation=tf.nn.leaky_relu)
        #self.dconv2 = Conv2DTranspose(64,3,1,'SAME',activation=tf.nn.leaky_relu)
        #self.dconv3 = Conv2DTranspose(64,3,1,'SAME',activation=tf.nn.leaky_relu)
        #self.dconv4 = Conv2DTranspose(64,5,2,'SAME',activation=tf.nn.leaky_relu)
        #self.dconv5 = Conv2DTranspose(64,3,1,'SAME',activation=tf.nn.leaky_relu)
        #self.dconv6 = Conv2DTranspose(64,3,1,'SAME',activation=tf.nn.leaky_relu)
        self.dconv7 = Conv2DTranspose(64,5,2,'SAME',activation=tf.nn.leaky_relu)
        self.dconv8 = Conv2DTranspose(1,5,2,'SAME',activation=tf.nn.leaky_relu)

    def call(self, x):
        #x = self.dconv1(x)
        #res = x
        #x = self.dconv2(x)
        #x = self.dconv3(x)
        #x = x + res
        #x = self.dconv4(x)
        #res = x
        #x = self.dconv5(x)
        #x = self.dconv6(x)
        #x = x + res
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
        return x


class ProClass:
    def __init__(self,model_class):
        self.models=[]
        for i in range(_CHANNELS):
            self.models.append(model_class())

    def load(self,path):
        for i in range(_CHANNELS):
            self.models[i].load_weights(path+str(i))


class Encoder(ProClass):
    def __init__(self):
        super(Encoder, self).__init__(BaseEncoder)

    def __call__(self, x):
        img_norm = x / 255

        img_channels = convert_to_colourspace(ycbcr_kernel, ycbcr_off, img_norm)

        encoded = []
        for i, img_channel in enumerate(img_channels):
            encoded.append(self.models[i](img_channel))

        encoded = tf.concat(encoded, axis=3)

        return np.round(encoded * 255)


class Decoder(ProClass):
    def __init__(self):
        super(Decoder, self).__init__(BaseDecoder)

    def __call__(self, x):
        img_norm = x / 255
        img_channels = tf.split(img_norm, 3, axis=3)

        decoded = []
        for i, img_channel in enumerate(img_channels):
            decoded.append(self.models[i](img_channel))

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

    def __call__(self, x, max_epochs, batch_size, entropy_loss_coef):

        optimizer_y = tf.keras.optimizers.Adam()
        optimizer_cbcr = tf.keras.optimizers.Adam()
        optimizer_entropy = tf.keras.optimizers.Adam()

        train_ds = tf.data.Dataset.from_tensor_slices(x).shuffle(10000).batch(batch_size)
        step=0
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
                    """
                    batch_encoded = tf.concat([encoded_0,encoded_1], axis=0)
                    aprox_entropy = self.entropy_model(batch_encoded)

                    encoded_denorm = batch_encoded * 255
                    transform_shape = tf.cast(
                        tf.cast(tf.shape(encoded_denorm), tf.float32) *
                        tf.convert_to_tensor([1, 4, 8, 1 / 32.0]), tf.int32)
                    img_to_convert = tf.reshape(
                        tf.cast(tf.round(encoded_denorm), tf.uint8),
                        transform_shape)
                    #img_to_convert=tf.concat(tf.split(img_to_convert,3,axis=0),axis=3)

                    size_list = tf.cast(
                        tf.stack(
                            tf.map_fn(get_png_size,
                                      img_to_convert,
                                      dtype=tf.int32)), tf.float32)
                    bpp = 8 * tf.reshape(size_list, (-1, 1)) / tf.cast(
                        tf.reduce_prod(tf.shape(img_to_convert)[1:3]),
                        tf.float32)
                    aprox_entropy_loss = tf.reduce_mean(
                        tf.keras.losses.MSE(bpp, aprox_entropy))

                    entropy_losses = [
                        tf.reduce_mean(chan_entr)
                        for chan_entr in tf.split(aprox_entropy, 3, axis=0)
                    ]

                    bpp_channels = tf.split(bpp, 3, axis=0)
                    """

                    decoded_0 = self.decoder_models[0](noisy_encoded_0)
                    ssim_0=tf.reduce_mean(tf.image.ssim(img_channels_0, decoded_0, max_val=1.0))
                    #loss_0=(1 - ssim_0) / 2 #+ entropy_loss_coef * entropy_losses[0]
                    loss_0=tf.reduce_mean((img_channels_0-decoded_0)**2)

                    decoded_1 = self.decoder_models[1](noisy_encoded_1)
                    ssim_1=tf.image.ssim(img_channels_1, decoded_1, max_val=1.0)
                    ssim_cb,ssim_cr=tf.split(ssim_1,2,axis=0)
                    ssim_cb=tf.reduce_mean(ssim_cb)
                    ssim_cr=tf.reduce_mean(ssim_cr)
                    ssim_1=tf.reduce_mean(ssim_1)
                    #loss_1=(1 - ssim_1) / 2 #+ entropy_loss_coef * tf.concat(entropy_losses[1:],axis=0)
                    loss_1=tf.reduce_mean((img_channels_1-decoded_1)**2)

                    ssims=[ssim_0.numpy(),ssim_cb.numpy(),ssim_cr.numpy()]
                    #bpp_res=[aux_bpp.numpy().mean() for aux_bpp in bpp_channels]
                    decoded = [decoded_0]+tf.split(decoded_1,2,axis=0)

                #with self.summary_writer.as_default():
                #    tf.summary.scalar('SSIM_Y', ssim_0,step=step)
                #    tf.summary.scalar('SSIM_Cb', ssim_cb,step=step)
                #    tf.summary.scalar('SSIM_Cr', ssim_cr,step=step)

                #print('EPOCH:', epoch, 'SSIM:', ssims, 'BPP:', bpp_res,'Entropy loss:', aprox_entropy_loss.numpy())
                print('EPOCH:', epoch, 'SSIM:', ssims)
                main_variables_y = self.encoder_models[
                    0].trainable_variables + self.decoder_models[
                        0].trainable_variables
                main_variables_cbcr = self.encoder_models[
                    1].trainable_variables + self.decoder_models[
                        1].trainable_variables

                #entropy_variables = self.entropy_model.trainable_variables

                gradients_y = y_tape.gradient(loss_0, main_variables_y)
                gradients_cbcr = cbcr_tape.gradient(loss_1, main_variables_cbcr)

                #gradients_entropy = entropy_tape.gradient(
                #    aprox_entropy_loss, entropy_variables)

                #optimizer_entropy.apply_gradients(
                #    zip(gradients_entropy, entropy_variables))
                optimizer_y.apply_gradients(zip(gradients_y, main_variables_y))
                optimizer_cbcr.apply_gradients(zip(gradients_cbcr, main_variables_cbcr))

                step+=1
                if step%10==0:
                    dec_out = tf.clip_by_value(convert_to_rgb(ycbcr_inv_kernel, ycbcr_off, *decoded), 0, 1).numpy()
                    dec_out=np.round(dec_out * 255).astype(np.uint8)
                    comparison_image=np.concatenate([(img_norm.numpy()[0]*255).astype(np.uint8),dec_out[0]],axis=1)
                    Image.fromarray(comparison_image).save('prueba.png')
                #    self._save()

    def _save(self):
        for i in range(_CHANNELS):
            self.encoder_models[i].save_weights('checkpoints/encoder{}'.format(i))
            self.decoder_models[i].save_weights('checkpoints/decoder{}'.format(i))

        print('checkpoint saved')


if __name__ == "__main__":
    imgs,_ = read_dataset('../data/imagenet_patches')
    training_obj = Training()
    training_obj(imgs, 15, 64, 0)

    encoder=Encoder()
    encoder.load('checkpoints/encoder')
    decoder=Decoder()
    decoder.load('checkpoints/decoder')

    enc_out=encoder(imgs[:10])
    print(enc_out.shape)
    dec_out=decoder(enc_out)
    print(dec_out.shape)
    print(imgs.max(),dec_out.max())