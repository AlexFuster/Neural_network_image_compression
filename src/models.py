from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose
import tensorflow as tf
from utils import convert_to_rgb, convert_to_colourspace,ycbcr_kernel, ycbcr_inv_kernel, ycbcr_off
import numpy as np

class BaseEncoder(tf.keras.Model):
  def __init__(self):
    super(BaseEncoder, self).__init__()
    self.conv1 = Conv2D(32,5,2,'SAME',activation='leaky_relu')
    self.conv2 = Conv2D(64,5,2,'SAME',activation='leaky_relu')
    self.conv3 = Conv2D(64,3,1,'SAME',activation='leaky_relu')
    self.conv4 = Conv2D(64,3,1,'SAME',activation='leaky_relu')
    self.conv5 = Conv2D(64,5,2,'SAME',activation='leaky_relu')
    self.conv6 = Conv2D(64,3,1,'SAME',activation='leaky_relu')
    self.conv7 = Conv2D(64,3,1,'SAME',activation='leaky_relu')
    self.conv8 = Conv2D(32,5,2,'SAME',activation='leaky_relu')

  def call(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    res=x
    x = self.conv3(x)
    x = self.conv4(x)
    x=x+res
    x = self.conv5(x)
    res=x
    x = self.conv6(x)
    x = self.conv7(x)
    x=x+res
    x = self.conv8(x)
    return tf.clip_by_value(x,0,1)

class BaseDecoder(tf.keras.Model):
  def __init__(self):
    super(BaseDecoder, self).__init__()
    self.dconv1 = Conv2DTranspose(64,5,2,'SAME',activation='leaky_relu')
    self.dconv2 = Conv2DTranspose(64,3,1,'SAME',activation='leaky_relu')
    self.dconv3 = Conv2DTranspose(64,3,1,'SAME',activation='leaky_relu')
    self.dconv4 = Conv2DTranspose(64,5,2,'SAME',activation='leaky_relu')
    self.dconv5 = Conv2DTranspose(64,3,1,'SAME',activation='leaky_relu')
    self.dconv6 = Conv2DTranspose(64,3,1,'SAME',activation='leaky_relu')
    self.dconv7 = Conv2DTranspose(64,5,2,'SAME',activation='leaky_relu')
    self.dconv8 = Conv2DTranspose(1,5,2,'SAME',activation='leaky_relu')

  def call(self, x):
    x = self.dconv1(x)
    res=x
    x = self.dconv2(x)
    x = self.dconv3(x)
    x=x+res
    x = self.dconv4(x)
    res=x
    x = self.dconv5(x)
    x = self.dconv6(x)
    x=x+res
    x = self.dconv7(x)
    x = self.dconv8(x)
    return tf.clip_by_value(x,0,1)


class Entropynet(tf.keras.Model):
  def __init__(self):
    super(Entropynet, self).__init__()
    self.conv1 = Conv2D(64,5,2,'SAME',activation='leaky_relu')
    self.conv2 = Conv2D(64,3,1,'SAME',activation='leaky_relu')
    self.conv3 = Conv2D(64,3,1,'SAME',activation='leaky_relu')
    self.flatten = Flatten()
    self.dense1=Dense(512)
    self.dense2=Dense(1)

  def call(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x=self.flatten(x)
    x=self.dense1(x)
    x=self.dense2(x)
    return x


class Encoder:
  def __init__(self,color=True):
    if color:
        channels=3
    else:
        channels=1

    self.models=[BaseEncoder() for i in range(channels)]
    self.color=color

  def call(self, x):
    img_norm=x/255
    if color:
        img_channels=convert_to_colourspace(ycbcr_kernel,ycbcr_off,img_norm)
    else:
        img_channels=(x,)
    encoded=[]
    for img_channel,model in zip(img_channels,self.models):
        encoded.append(model(img_channel))

    encoded=tf.concat(encoded,axis=3)

    return np.round(encoded*255)

class Decoder:
  def __init__(self,color=True):
    if color:
        channels=3
    else:
        channels=1

    self.models=[BaseDecoder() for i in range(channels)]
    self.color=color

  def call(self, x):
    img_norm=x/255
    if color:
        img_channels=tf.split(self.img_norm,3,axis=3)
    else:
        img_channels=(x,)
    decoded=[]
    for img_channel,model in zip(img_channels,self.models):
        decoded.append(model(img_channel))

    decoded=tf.clip_by_value(convert_to_rgb(ycbcr_inv_kernel,ycbcr_off,decoded),0,1)

    return np.round(decoded*255)

class Training(tf.keras.Model):
  def __init__(self,color=True):
    super(Training, self).__init__()
    if color:
        channels=3
    else:
        channels=1

    self.encoder_models=[BaseDecoder() for i in range(channels)]
    self.decoder_models=[BaseDecoder() for i in range(channels)]
    self.entropy_model=Entropynet()
    self.color=color

  def call(self,x,max_epochs,batch_size):
    optimizer_y = tf.keras.optimizers.Adam()
    optimizer_cbcr = tf.keras.optimizers.Adam()

    train_ds = tf.data.Dataset.from_tensor_slices(x).shuffle(10000).batch(batch_size)

    for epoch in range(max_epochs):
        for images in train_ds:
            with tf.GradientTape() as tape:
                img_norm=x/255
                img_norm=tf.image.random_flip_left_right(img_norm)
                img_norm=tf.image.random_flip_up_down(img_norm)
                img_channels=convert_to_colourspace(ycbcr_kernel,ycbcr_off,img_norm)
                encoded=[]
                noisy_encoded_channels=[]
                for img_channel,model in zip(img_channels,self.encoder_models):
                    aux_out=model(img_channel)
                    encoded.append(aux_out)
                    noisy_encoded_channels.append(tf.clip_by_value(aux_out+tf.random_uniform(tf.shape(aux_out),-0.5,0.5)/255,0,1))

                batch_encoded=tf.concat(encoded,axis=0)
                aprox_entropy=self.entropy_model(batch_encoded)
                aprox_entropy=tf.split(aprox_entropy,3,axis=0)

                entropy_losses=tf.reduce_mean(aprox_entropy,axis=3)

                decoded=[]
                ssims=0
                losses=[]
                for entropy_loss,img_channel,noisy_encoded_channel,model in zip(entropy_losses,img_channels,noisy_encoded_channels,self.decoder_models):
                    aux_out=model(noisy_encoded_channel)
                    decoded.append(aux_out)
                    ssim=tf.reduce_mean(tf.image.ssim_multiscale(img_channel,aux_out,max_val=1.0))
                    ssims+=ssim
                    losses.append((1-ssim)/2 + entropy_loss_coef*entropy_loss)

            gradients_y = tape.gradient(losses[0], model.trainable_variables)
            gradients_cbcr = tape.gradient(losses[1], model.trainable_variables)

            optimizer_y.apply_gradients(zip(gradients_y, model.trainable_variables))
            optimizer_cbcr.apply_gradients(zip(gradients_cbcr, model.trainable_variables))
