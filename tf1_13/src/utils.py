import numpy as np
import tensorflow as tf
import math
import os
from PIL import Image
import pickle

ycbcr_kernel=np.array([[0.299,0.587,0.114],[-0.16874,-0.33126,0.5],[0.5,-0.41869,-0.08131]])
ycbcr_inv_kernel=np.linalg.inv(ycbcr_kernel)
ycbcr_off=np.array([0,0.5,0.5])

pca_kernel=np.array([[1/3,1/3,1/3],[-0.5,0,0.5],[0.25,-0.5,0.25]])
pca_inv_kernel=np.linalg.inv(pca_kernel)
pca_off=np.array([0,0.5,0.5])


def prelu(_x,name):

    alphas = tf.get_variable(name, _x.get_shape()[-1],
                        initializer=tf.constant_initializer(0.2),
                            dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg

def downsample_entropy(img):
    layer=tf.layers.conv2d(img,64,(5,5),activation=tf.nn.leaky_relu,strides=(2,2),name='econv1',trainable=False,reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d(layer,64,(3,3),padding="SAME",activation=tf.nn.leaky_relu,name='econv2',trainable=False,reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d(layer,64,(3,3),padding="SAME",activation=tf.nn.leaky_relu,name='econv3',trainable=False,reuse=tf.AUTO_REUSE)
    return layer

def dense_entropy(img):
    layer=tf.layers.flatten(img)
    layer=tf.layers.dense(layer,512,activation=tf.nn.leaky_relu,name='dense_2',trainable=False,reuse=tf.AUTO_REUSE)
    layer=tf.layers.dense(layer,1,name='dense_3',trainable=False,reuse=tf.AUTO_REUSE)
    return layer

def encode_res(img,chanel):
    layer=tf.layers.conv2d(img,32,(5,5),strides=(2,2),padding="SAME",name='conv1{}'.format(chanel),reuse=tf.AUTO_REUSE)
    layer=prelu(layer,'cprelu_1{}'.format(chanel))
    layer=tf.layers.conv2d(layer,64,(5,5),strides=(2,2),padding="SAME",name='conv2{}'.format(chanel),reuse=tf.AUTO_REUSE)
    layer=prelu(layer,'cprelu_2{}'.format(chanel))
    res=tf.layers.conv2d(layer,64,(3,3),padding="SAME",name='conv3{}'.format(chanel),reuse=tf.AUTO_REUSE)
    res=prelu(res,'cprelu_3{}'.format(chanel))
    res=tf.layers.conv2d(res,64,(3,3),padding="SAME",name='conv4{}'.format(chanel),reuse=tf.AUTO_REUSE)
    layer=res+layer
    layer=prelu(layer,'cprelu_4{}'.format(chanel))
    layer=tf.layers.conv2d(layer,64,(5,5),strides=(2,2),padding="SAME",name='conv5{}'.format(chanel),reuse=tf.AUTO_REUSE)
    layer=prelu(layer,'cprelu_5{}'.format(chanel))
    res=tf.layers.conv2d(layer,64,(3,3),padding="SAME",name='conv6{}'.format(chanel),reuse=tf.AUTO_REUSE)
    res=prelu(res,'cprelu_6{}'.format(chanel))
    res=tf.layers.conv2d(res,64,(3,3),padding="SAME",name='conv7{}'.format(chanel),reuse=tf.AUTO_REUSE)
    layer=res+layer
    layer=prelu(layer,'cprelu_7{}'.format(chanel))
    layer=tf.layers.conv2d(layer,32,(5,5),strides=(2,2),padding="SAME",name='conv8{}'.format(chanel),reuse=tf.AUTO_REUSE)
    layer=prelu(layer,'cprelu_8{}'.format(chanel))
    return layer

def decode_res(encoded,chanel):
    layer=tf.layers.conv2d_transpose(encoded,64,(5,5),padding="SAME",strides=(2,2),name='dconv1{}'.format(chanel),reuse=tf.AUTO_REUSE)
    layer=prelu(layer,'dprelu_1{}'.format(chanel))
    res=tf.layers.conv2d(layer,64,(3,3),padding="SAME",name='dconv2{}'.format(chanel),reuse=tf.AUTO_REUSE)
    res=prelu(res,'dprelu_2{}'.format(chanel))
    res=tf.layers.conv2d(res,64,(3,3),padding="SAME",name='dconv3{}'.format(chanel),reuse=tf.AUTO_REUSE)
    layer=res+layer
    layer=prelu(layer,'dprelu_3{}'.format(chanel))
    layer=tf.layers.conv2d_transpose(layer,64,(5,5),padding="SAME",strides=(2,2),name='dconv4{}'.format(chanel),reuse=tf.AUTO_REUSE)
    layer=prelu(layer,'dprelu_4{}'.format(chanel))
    res=tf.layers.conv2d(layer,64,(3,3),padding="SAME",name='dconv5{}'.format(chanel),reuse=tf.AUTO_REUSE)
    res=prelu(res,'dprelu_5{}'.format(chanel))
    res=tf.layers.conv2d(res,64,(3,3),padding="SAME",name='dconv6{}'.format(chanel),reuse=tf.AUTO_REUSE)
    layer=res+layer
    layer=prelu(layer,'dprelu_6{}'.format(chanel))
    layer=tf.layers.conv2d_transpose(layer,64,(5,5),padding="SAME",strides=(2,2),name='dconv7{}'.format(chanel),reuse=tf.AUTO_REUSE)
    layer=prelu(layer,'dprelu_7{}'.format(chanel))
    layer=tf.layers.conv2d_transpose(layer,1,(5,5),padding="SAME",strides=(2,2),name='dconv8{}'.format(chanel),reuse=tf.AUTO_REUSE)
    layer=prelu(layer,'dprelu_8{}'.format(chanel))
    return layer
    
def _project(kernel,tensor_0,tensor_1,tensor_2):
    out_0=tensor_0*kernel[0,0]+tensor_1*kernel[0,1]+tensor_2*kernel[0,2]
    out_1=tensor_0*kernel[1,0]+tensor_1*kernel[1,1]+tensor_2*kernel[1,2]
    out_2=tensor_0*kernel[2,0]+tensor_1*kernel[2,1]+tensor_2*kernel[2,2]
    return out_0,out_1,out_2

def convert_to_rgb(kernel,offsets,tensor_0,tensor_1,tensor_2):
    out_0,out_1,out_2=_project(kernel,tensor_0-offsets[0],tensor_1-offsets[1],tensor_2-offsets[2])
    return tf.concat([out_0,out_1,out_2],axis=3)

def convert_to_colourspace(kernel,offsets,tensor):
    out_0,out_1,out_2=tf.split(tensor,3,axis=3)
    out_0,out_1,out_2=_project(kernel,out_0,out_1,out_2)
    return out_0+offsets[0],out_1+offsets[1],out_2+offsets[2]


def encode(img, chanel):
    encoded=encode_res(img, chanel)
    return encoded
         
def decode(encoded, chanel):
    decoded=decode_res(encoded, chanel)
    return decoded


def get_next_log_name(log_dir,lastdir=False):
    try:
        return log_dir+str(max([int(i.split('_')[0]) for i in os.listdir(log_dir)])+1-int(lastdir))
    except ValueError:
        return log_dir+'1'

def save_imag(img,output_dir,filename):
    Image.fromarray(img).save(output_dir+'/'+filename+'.png')

def save_compressed(img,output_dir,filename):
    assert (np.round(img)-img).sum()==0
    Image.fromarray(img).save(output_dir+'/'+filename+'.png',optimize=True)

def read_dataset(dataset_path,istrain=False):
      
    imgs=[]
    filenames=[]
    for w in sorted(os.listdir(dataset_path)):
        if w.split('.')[-1] in ['png','jpg','jpeg','gif','pgm','ppm','bmp','jp2']:             
            with Image.open(dataset_path+'/'+w) as aux_im:
                aux_im_arr=np.array(aux_im)
                if len(aux_im_arr.shape)==3:
                    imgs.append(aux_im_arr)
                    filenames.append('.'.join(w.split('.')[:-1]))
    
    imgs=np.array(imgs)
    print(imgs.shape)
    
    if len(imgs.shape)==1:
        for i,img in enumerate(imgs):
            if len(img.shape)==2:
                imgs[i]=img.reshape((1,)+img.shape+(1,))
            else:
                imgs[i]=img.reshape((1,)+img.shape)

            imgs[i]=imgs[i].astype(np.uint8)

            
    elif len(imgs.shape)==3:
        imgs=imgs.reshape(imgs.shape+(1,)).astype(np.uint8)

    elif len(imgs.shape)==4:
        imgs=imgs.astype(np.uint8)

    return imgs, filenames