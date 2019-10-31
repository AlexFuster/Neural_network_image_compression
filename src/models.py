from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose
import tensorflow as tf
from utils import convert_to_rgb, convert_to_colourspace, ycbcr_kernel, ycbcr_inv_kernel, ycbcr_off, read_dataset
import numpy as np
import pickle

get_png_size = lambda xin: tf.strings.length(tf.image.encode_png(xin))
_CHANNELS = 3


def save(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load(path):
    with open(path, 'rb') as f:
        model=pickle.load(f)
    return model

class BaseEncoder(tf.keras.Model):
    def __init__(self):
        super(BaseEncoder, self).__init__()
        self.conv1 = Conv2D(32, 5, 2, 'SAME', activation=tf.nn.leaky_relu)
        self.conv2 = Conv2D(64, 5, 2, 'SAME', activation=tf.nn.leaky_relu)
        self.conv3 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        self.conv4 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        self.conv5 = Conv2D(64, 5, 2, 'SAME', activation=tf.nn.leaky_relu)
        self.conv6 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        self.conv7 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        self.conv8 = Conv2D(32, 5, 2, 'SAME', activation=tf.nn.leaky_relu)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        res = x
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + res
        x = self.conv5(x)
        res = x
        x = self.conv6(x)
        x = self.conv7(x)
        x = x + res
        x = self.conv8(x)
        return tf.clip_by_value(x, 0, 1)


class BaseDecoder(tf.keras.Model):
    def __init__(self):
        super(BaseDecoder, self).__init__()
        self.dconv1 = Conv2DTranspose(64,5,2,'SAME',activation=tf.nn.leaky_relu)
        self.dconv2 = Conv2DTranspose(64,3,1,'SAME',activation=tf.nn.leaky_relu)
        self.dconv3 = Conv2DTranspose(64,3,1,'SAME',activation=tf.nn.leaky_relu)
        self.dconv4 = Conv2DTranspose(64,5,2,'SAME',activation=tf.nn.leaky_relu)
        self.dconv5 = Conv2DTranspose(64,3,1,'SAME',activation=tf.nn.leaky_relu)
        self.dconv6 = Conv2DTranspose(64,3,1,'SAME',activation=tf.nn.leaky_relu)
        self.dconv7 = Conv2DTranspose(64,5,2,'SAME',activation=tf.nn.leaky_relu)
        self.dconv8 = Conv2DTranspose(1,5,2,'SAME',activation=tf.nn.leaky_relu)

    def call(self, x):
        x = self.dconv1(x)
        res = x
        x = self.dconv2(x)
        x = self.dconv3(x)
        x = x + res
        x = self.dconv4(x)
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
        return x


class Encoder:
    def __init__(self, models):
        self.models = models

    def __call__(self, x):
        img_norm = x / 255

        img_channels = convert_to_colourspace(ycbcr_kernel, ycbcr_off,
                                              img_norm)

        encoded = []
        for i, img_channel in enumerate(img_channels):
            encoded.append(self.models[i](img_channel))

        encoded = tf.concat(encoded, axis=3)

        return np.round(encoded * 255)


class Decoder:
    def __init__(self, models):
        self.models = models

    def __call__(self, x):
        img_norm = x / 255
        img_channels = tf.split(self.img_norm, 3, axis=3)

        decoded = []
        for i, img_channel in enumerate(img_channels):
            decoded.append(self.models[i](img_channel))

        decoded = tf.clip_by_value(
            convert_to_rgb(ycbcr_inv_kernel, ycbcr_off, decoded), 0, 1)

        return np.round(decoded * 255)


class Training:
    def __init__(self):

        self.encoder_models = []
        self.decoder_models = []
        self.epoch = 0

        for i in range(_CHANNELS):
            self.encoder_models.append(BaseEncoder())
            self.decoder_models.append(BaseDecoder())

        self.entropy_model = Entropynet()

    def __call__(self, x, max_epochs, batch_size, entropy_loss_coef):

        optimizer_y = tf.keras.optimizers.Adam()
        optimizer_cb = tf.keras.optimizers.Adam()
        optimizer_cr = tf.keras.optimizers.Adam()
        optimizer_entropy = tf.keras.optimizers.Adam()

        train_ds = tf.data.Dataset.from_tensor_slices(x).shuffle(10000).batch(batch_size)

        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch
            for images in train_ds:
                with tf.GradientTape() as y_tape, tf.GradientTape(
                ) as cb_tape, tf.GradientTape() as cr_tape, tf.GradientTape(
                ) as entropy_tape:
                    img_norm = images / 255
                    img_norm = tf.image.random_flip_left_right(img_norm)
                    img_norm = tf.image.random_flip_up_down(img_norm)
                    img_channels = convert_to_colourspace(
                        ycbcr_kernel, ycbcr_off, img_norm)
                    encoded = []
                    noisy_encoded_channels = []
                    for i, img_channel in enumerate(img_channels):
                        aux_out = self.encoder_models[i](img_channel)
                        encoded.append(aux_out)
                        noisy_encoded_channels.append(
                            tf.clip_by_value(
                                aux_out + tf.random.uniform(
                                    tf.shape(aux_out), -0.5, 0.5) / 255, 0, 1))

                    batch_encoded = tf.concat(encoded, axis=0)
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

                    decoded = []
                    ssims = []
                    losses = []
                    bpp_res = []
                    bpp_channels = tf.split(bpp, 3, axis=0)

                    for i, img_channel in enumerate(img_channels):
                        aux_out = self.decoder_models[i](
                            noisy_encoded_channels[i])
                        decoded.append(aux_out)
                        ssim = tf.reduce_mean(
                            tf.image.ssim(img_channel, aux_out, max_val=1.0))
                        ssims.append(ssim.numpy())
                        bpp_res.append(bpp_channels[i].numpy().mean())
                        losses.append((1 - ssim) / 2 +
                                      entropy_loss_coef * entropy_losses[i])

                print('EPOCH:', epoch, 'SSIM:', ssims, 'BPP:', bpp_res,
                      'Entropy loss:', aprox_entropy_loss.numpy())

                main_variables_y = self.encoder_models[
                    0].trainable_variables + self.decoder_models[
                        0].trainable_variables
                main_variables_cb = self.encoder_models[
                    1].trainable_variables + self.decoder_models[
                        1].trainable_variables
                main_variables_cr = self.encoder_models[
                    2].trainable_variables + self.decoder_models[
                        2].trainable_variables

                entropy_variables = self.entropy_model.trainable_variables

                gradients_y = y_tape.gradient(losses[0], main_variables_y)
                gradients_cb = cb_tape.gradient(losses[1], main_variables_cb)
                gradients_cr = cr_tape.gradient(losses[2], main_variables_cr)

                gradients_entropy = entropy_tape.gradient(
                    aprox_entropy_loss, entropy_variables)

                optimizer_entropy.apply_gradients(
                    zip(gradients_entropy, entropy_variables))
                optimizer_y.apply_gradients(zip(gradients_y, main_variables_y))
                optimizer_cb.apply_gradients(
                    zip(gradients_cb, main_variables_cb))
                optimizer_cr.apply_gradients(
                    zip(gradients_cr, main_variables_cr))

                self._save()

    def _save(self):
        encoder=Encoder(self.encoder_models)
        decoder=Decoder(self.decoder_models)
        save(self,'training.pkl')
        save(encoder,'encoder.pkl')
        save(decoder,'decoder.pkl')
        print('checkpoint saved')


if __name__ == "__main__":
    imgs,_ = read_dataset('../data/imagenet_patches')
    training_obj = Training()
    training_obj(imgs, 1, 64, 0.0005)
    encoder=load('encoder.pkl')
    decoder=load('decoder.pkl')
    enc_out=encoder(imgs[:10])
    print(enc_out.numpy().shape)
    dec_out=decoder(enc_out)
    print(dec_out.numpy().shape)
