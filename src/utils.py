import numpy as np
#import pandas as pd
import tensorflow as tf
import math
import os
#from sklearn.utils import resample
#from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.decomposition import PCA
import pickle

def downsample_entropy(img):
    layer=tf.layers.conv2d(img,32,(5,5),activation=tf.nn.relu,name='econv1',trainable=False,reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d(layer,32,(3,3),activation=tf.nn.relu,name='econv2',trainable=False,reuse=tf.AUTO_REUSE)
    layer=tf.layers.max_pooling2d(layer,(2,2),2,name='pool1')
    layer=tf.layers.conv2d(layer,32,(3,3),activation=tf.nn.relu,name='econv3',trainable=False,reuse=tf.AUTO_REUSE)
    return layer

def dense_entropy(img):
    layer=tf.layers.flatten(img)
    layer=tf.layers.dense(layer,512,activation=tf.nn.relu,name='dense_1',trainable=False,reuse=tf.AUTO_REUSE)
    layer=tf.layers.dense(layer,512,activation=tf.nn.relu,name='dense_2',trainable=False,reuse=tf.AUTO_REUSE)
    layer=tf.layers.dense(layer,1,activation=tf.nn.relu,name='dense_3',trainable=False,reuse=tf.AUTO_REUSE)
    return layer

def encode_with_pool(img):
    layer=tf.layers.conv2d(img,16,(5,5),activation=tf.nn.relu,name='conv1',reuse=tf.AUTO_REUSE)
    layer=tf.layers.average_pooling2d(layer,(2,2),2,name='pool1')
    layer=tf.layers.conv2d(layer,8,(3,3),activation=tf.nn.relu,name='conv2',reuse=tf.AUTO_REUSE)
    layer=tf.layers.average_pooling2d(layer,(2,2),2,name='pool2')
    layer=tf.layers.conv2d(layer,8,(3,3),activation=tf.nn.relu,name='conv3',reuse=tf.AUTO_REUSE)
    return layer

def encode_pool_nrom(img):
    layer=tf.layers.conv2d(img,16,(5,5),activation=tf.nn.relu,name='conv1',reuse=tf.AUTO_REUSE)
    layer=tf.layers.max_pooling2d(layer,(2,2),2,name='pool1')
    layer=tf.layers.batch_normalization(layer)
    layer=tf.layers.conv2d(layer,8,(3,3),activation=tf.nn.relu,name='conv2',reuse=tf.AUTO_REUSE)
    layer=tf.layers.max_pooling2d(layer,(2,2),2,name='pool2')
    layer=tf.layers.batch_normalization(layer)
    layer=tf.layers.conv2d(layer,8,(3,3),activation=tf.nn.relu,name='conv3',reuse=tf.AUTO_REUSE)
    return layer

def encode_no_pool_432(img):
    layer=tf.layers.conv2d(img,16,(4,4),strides=(2,2),activation=tf.nn.relu,name='conv1',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d(layer,8,(3,3),strides=(2,2),activation=tf.nn.relu,name='conv2',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d(layer,8,(2,2),strides=(2,2),activation=tf.nn.relu,name='conv3',reuse=tf.AUTO_REUSE)    
    return layer

def encode_no_pool_855(img):
    layer=tf.layers.conv2d(img,16,(8,8),strides=(1,1),activation=tf.nn.relu,name='conv1',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d(layer,8,(5,5),strides=(2,2),activation=tf.nn.relu,name='conv2',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d(layer,8,(5,5),strides=(2,2),activation=tf.nn.relu,name='conv3',reuse=tf.AUTO_REUSE)    
    return layer

def encode_no_pool_533(img): #result 4*4*8
    layer=tf.layers.conv2d(img,16,(5,5),strides=(2,2),padding="SAME",activation=tf.nn.relu,name='conv1',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d(layer,8,(3,3),strides=(2,2),activation=tf.nn.relu,name='conv2',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d(layer,8,(3,3),activation=tf.nn.relu,name='conv3',reuse=tf.AUTO_REUSE)    
    return layer

def decode_deconv_335(encoded):
    layer=tf.layers.conv2d_transpose(encoded,8,(3,3),activation=tf.nn.relu,name='dconv1',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d_transpose(layer,16,(3,3),strides=(2,2),activation=tf.nn.relu,name='dconv2',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d_transpose(layer,1,(5,5),strides=(2,2),activation=tf.nn.relu,name='dconv3',reuse=tf.AUTO_REUSE)
    layer=tf.image.resize_images(layer,(28,28))
    return layer

def decode_deconv_234(encoded):
    layer=tf.layers.conv2d_transpose(encoded,8,(2,2),strides=(2,2),activation=tf.nn.relu,name='dconv1',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d_transpose(layer,16,(3,3),strides=(2,2),activation=tf.nn.relu,name='dconv2',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d_transpose(layer,1,(4,4),strides=(2,2),activation=tf.nn.relu,name='dconv3',reuse=tf.AUTO_REUSE)
    return layer

def decode_deconv_558(encoded):
    layer=tf.layers.conv2d_transpose(encoded,8,(5,5),strides=(2,2),activation=tf.nn.relu,name='dconv1',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d_transpose(layer,16,(5,5),strides=(2,2),activation=tf.nn.relu,name='dconv2',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d_transpose(layer,1,(8,8),strides=(1,1),activation=tf.nn.relu,name='dconv3',reuse=tf.AUTO_REUSE)
    return layer

def decode_558_norm(encoded):
    layer=tf.layers.conv2d_transpose(encoded,8,(5,5),strides=(2,2),activation=tf.nn.relu,name='dconv1',reuse=tf.AUTO_REUSE)
    layer=tf.layers.batch_normalization(layer)
    layer=tf.layers.conv2d_transpose(layer,16,(5,5),strides=(2,2),activation=tf.nn.relu,name='dconv2',reuse=tf.AUTO_REUSE)
    layer=tf.layers.batch_normalization(layer)    
    layer=tf.layers.conv2d_transpose(layer,1,(8,8),strides=(1,1),activation=tf.nn.relu,name='dconv3',reuse=tf.AUTO_REUSE)
    return layer

def double_img(img):
    shape=tf.shape(img)
    return 2*shape[1],2*shape[2]

def decode_resize(encoded,method):
    layer=tf.layers.conv2d_transpose(encoded,8,(3,3),activation=tf.nn.relu,name='dconv1',reuse=tf.AUTO_REUSE)
    layer=tf.image.resize_images(layer,double_img(layer),method=method)
    layer=tf.layers.conv2d_transpose(encoded,16,(3,3),activation=tf.nn.relu,name='dconv1',reuse=tf.AUTO_REUSE)
    layer=tf.image.resize_images(layer,double_img(layer),method=method)
    layer=tf.layers.conv2d_transpose(encoded,1,(5,5),activation=tf.nn.relu,name='dconv1',reuse=tf.AUTO_REUSE)
    return layer

def decode_resize_norm(encoded,method):
    layer=tf.layers.batch_normalization(encoded)
    layer=tf.layers.conv2d_transpose(layer,8,(3,3),activation=tf.nn.relu,name='dconv1',reuse=tf.AUTO_REUSE)
    layer=tf.layers.batch_normalization(layer)
    layer=tf.image.resize_images(layer,double_img(layer),method=method)
    layer=tf.layers.conv2d_transpose(encoded,16,(3,3),activation=tf.nn.relu,name='dconv1',reuse=tf.AUTO_REUSE)
    layer=tf.layers.batch_normalization(layer)
    layer=tf.image.resize_images(layer,double_img(layer),method=method)
    layer=tf.layers.conv2d_transpose(encoded,1,(5,5),activation=tf.nn.relu,name='dconv1',reuse=tf.AUTO_REUSE)
    return layer

def encode_no_pool_gdn_432(img):
    layer=tf.layers.conv2d(img,16,(4,4),strides=(2,2),name='conv1',reuse=tf.AUTO_REUSE)
    layer=tf.contrib.layers.gdn(layer,name="gdn1",reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d(layer,8,(3,3),strides=(2,2),activation=tf.nn.relu,name='conv2',reuse=tf.AUTO_REUSE)
    layer=tf.contrib.layers.gdn(layer,name="gdn2",reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d(layer,8,(2,2),strides=(2,2),activation=tf.nn.relu,name='conv3',reuse=tf.AUTO_REUSE)    
    layer=tf.contrib.layers.gdn(layer,name="gdn3",reuse=tf.AUTO_REUSE)
    return layer

def decode_deconv_gdn_234(encoded):
    layer=tf.layers.conv2d_transpose(encoded,8,(2,2),strides=(2,2),activation=tf.nn.relu,name='dconv1',reuse=tf.AUTO_REUSE)
    layer=tf.contrib.layers.gdn(layer,inverse=True,name="igdn1",reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d_transpose(layer,16,(3,3),strides=(2,2),activation=tf.nn.relu,name='dconv2',reuse=tf.AUTO_REUSE)
    layer=tf.contrib.layers.gdn(layer,inverse=True,name="igdn2",reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d_transpose(layer,1,(4,4),strides=(2,2),activation=tf.nn.relu,name='dconv3',reuse=tf.AUTO_REUSE)
    layer=tf.contrib.layers.gdn(layer,inverse=True,name="igdn3",reuse=tf.AUTO_REUSE)
    return layer

def phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    c=int(c/(r*r))
    X = tf.reshape(I, (-1, a, b, r, r, c))
    X = tf.transpose(X, (0, 1, 2, 4, 3, 5))  # bsize, a, b, r, r, c
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r, c]
    X = tf.concat([tf.squeeze(x) for x in X],2)  # bsize, b, a*r, r, c
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r, c]
    X = tf.concat([tf.squeeze(x) for x in X],2)  #bsize, a*r, b*r, c
    return tf.reshape(X, (-1, a*r, b*r, c))

def conv2d_subpix(img,c,r,k,name):
    layer=tf.layers.conv2d(img,c*r*r,(k,k),activation=tf.nn.relu,name=name,reuse=tf.AUTO_REUSE)
    return phase_shift(layer,r)

def decode_subpix(encoded): #Pendiente de revision
    layer=conv2d_subpix(encoded,8,3,2,"subpix1")
    layer=conv2d_subpix(layer,16,2,2,"subpix2")
    layer=conv2d_subpix(layer,8,2,3,"subpix3")
    layer=conv2d_subpix(layer,1,2,3,"subpix4")
    return layer


def encode_res(img):
    layer=tf.layers.conv2d(img,32,(5,5),strides=(2,2),padding="SAME",activation=tf.nn.relu,name='conv1',reuse=tf.AUTO_REUSE)
    res=tf.layers.conv2d(layer,32,(3,3),padding="SAME",activation=tf.nn.relu,name='conv2',reuse=tf.AUTO_REUSE)
    res=tf.layers.conv2d(res,32,(3,3),padding="SAME",activation=tf.nn.relu,name='conv3',reuse=tf.AUTO_REUSE)
    layer=res+layer
    layer=tf.layers.conv2d(layer,32,(5,5),strides=(2,2),activation=tf.nn.relu,name='conv4',reuse=tf.AUTO_REUSE)
    res=tf.layers.conv2d(layer,32,(3,3),padding="SAME",activation=tf.nn.relu,name='conv5',reuse=tf.AUTO_REUSE)
    res=tf.layers.conv2d(res,32,(3,3),padding="SAME",activation=tf.nn.relu,name='conv6',reuse=tf.AUTO_REUSE)
    layer=res+layer
    layer=tf.layers.conv2d(layer,32,(5,5),strides=(2,2),activation=tf.nn.relu,name='conv7',reuse=tf.AUTO_REUSE)
    res=tf.layers.conv2d(layer,32,(3,3),padding="SAME",activation=tf.nn.relu,name='conv8',reuse=tf.AUTO_REUSE)
    res=tf.layers.conv2d(res,32,(3,3),padding="SAME",activation=tf.nn.relu,name='conv9',reuse=tf.AUTO_REUSE)
    layer=res+layer
    layer=tf.layers.conv2d(layer,32,(5,5),strides=(2,2),activation=tf.nn.relu,name='conv10',reuse=tf.AUTO_REUSE)
    return layer
    
def decode_res(encoded,shape):
    layer=tf.layers.conv2d_transpose(encoded,32,(5,5),strides=(2,2),activation=tf.nn.relu,name='dconv1',reuse=tf.AUTO_REUSE)
    res=tf.layers.conv2d_transpose(layer,32,(3,3),padding="SAME",activation=tf.nn.relu,name='dconv2',reuse=tf.AUTO_REUSE)
    res=tf.layers.conv2d_transpose(res,32,(3,3),padding="SAME",activation=tf.nn.relu,name='dconv3',reuse=tf.AUTO_REUSE)
    layer=res+layer
    layer=tf.layers.conv2d_transpose(layer,32,(5,5),strides=(2,2),activation=tf.nn.relu,name='dconv4',reuse=tf.AUTO_REUSE)
    res=tf.layers.conv2d_transpose(layer,32,(3,3),padding="SAME",activation=tf.nn.relu,name='dconv5',reuse=tf.AUTO_REUSE)
    res=tf.layers.conv2d_transpose(res,32,(3,3),padding="SAME",activation=tf.nn.relu,name='dconv6',reuse=tf.AUTO_REUSE)
    layer=res+layer
    layer=tf.layers.conv2d_transpose(layer,32,(5,5),strides=(2,2),activation=tf.nn.relu,name='dconv7',reuse=tf.AUTO_REUSE)
    res=tf.layers.conv2d_transpose(layer,32,(3,3),padding="SAME",activation=tf.nn.relu,name='dconv8',reuse=tf.AUTO_REUSE)
    res=tf.layers.conv2d_transpose(res,32,(3,3),padding="SAME",activation=tf.nn.relu,name='dconv9',reuse=tf.AUTO_REUSE)
    layer=res+layer
    layer=tf.layers.conv2d_transpose(layer,1,(5,5),strides=(2,2),activation=tf.nn.relu,name='dconv10',reuse=tf.AUTO_REUSE)
    if shape is not None:
        layer=tf.image.resize_images(layer,(shape[0],shape[1]))
    return layer

def encode_deep(img):
    layer=tf.layers.conv2d(img,32,(5,5),strides=(2,2),padding="SAME",activation=tf.nn.relu,name='conv1',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d(layer,32,(3,3),activation=tf.nn.relu,name='conv2',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d(layer,64,(5,5),strides=(2,2),activation=tf.nn.relu,name='conv3',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d(layer,64,(3,3),activation=tf.nn.relu,name='conv4',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d(layer,64,(3,3),activation=tf.nn.relu,name='conv5',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d(layer,64,(5,5),strides=(2,2),activation=tf.nn.relu,name='conv6',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d(layer,32,(3,3),activation=tf.nn.relu,name='conv7',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d(layer,32,(5,5),strides=(2,2),activation=tf.nn.relu,name='conv8',reuse=tf.AUTO_REUSE)
    #layer=tf.layers.conv2d(layer,32,(3,3),activation=tf.nn.relu,name='conv9',reuse=tf.AUTO_REUSE)
    #layer=tf.layers.conv2d(layer,32,(3,3),activation=tf.nn.relu,name='conv10',reuse=tf.AUTO_REUSE)
    #layer=tf.layers.conv2d(layer,32,(5,5),strides=(2,2),activation=tf.nn.relu,name='conv11',reuse=tf.AUTO_REUSE)
    return layer

def decode_deep(encoded,shape):
    #layer=tf.layers.conv2d_transpose(encoded,32,(5,5),strides=(2,2),activation=tf.nn.relu,name='dconv1',reuse=tf.AUTO_REUSE)
    #layer=tf.layers.conv2d_transpose(layer,32,(3,3),activation=tf.nn.relu,name='dconv2',reuse=tf.AUTO_REUSE)
    #layer=tf.layers.conv2d_transpose(layer,32,(3,3),activation=tf.nn.relu,name='dconv3',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d_transpose(encoded,32,(5,5),strides=(2,2),activation=tf.nn.relu,name='dconv1',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d_transpose(layer,64,(3,3),activation=tf.nn.relu,name='dconv2',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d_transpose(layer,64,(5,5),strides=(2,2),activation=tf.nn.relu,name='dconv3',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d_transpose(layer,64,(3,3),activation=tf.nn.relu,name='dconv4',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d_transpose(layer,64,(3,3),activation=tf.nn.relu,name='dconv5',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d_transpose(layer,32,(5,5),strides=(2,2),activation=tf.nn.relu,name='dconv6',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d_transpose(layer,32,(3,3),activation=tf.nn.relu,name='dconv7',reuse=tf.AUTO_REUSE)
    layer=tf.layers.conv2d_transpose(layer,1,(5,5),strides=(2,2),activation=tf.nn.relu,name='dconv8',reuse=tf.AUTO_REUSE)
    if shape is not None:
        layer=tf.image.resize_images(layer,(shape[0],shape[1])) 
    return layer


def encode(img,config=''):
    if config=='mnist':
        encoded=encode_no_pool_533(img)
    else:
        encoded=encode_res(img)
    encoded=tf.clip_by_value(encoded,0,1)
    return encoded
         
def decode(encoded,shape=None,config=''):
    if config=='mnist':
        decoded=decode_deconv_335(encoded)
    else:
        decoded=decode_res(encoded,shape) #add shape if decode_deep
    decoded=tf.clip_by_value(decoded,0,1)
    return decoded


def get_next_log_name(log_dir,lastdir=False):
    try:
        return log_dir+str(max([int(i.split('_')[0]) for i in os.listdir(log_dir)])+1-int(lastdir))
    except ValueError:
        return log_dir+'1'
    


def code_array(X,max_val):
    if max_val <=255:
        return X.astype(np.uint8)
    elif max_val <=65535:
        return X.astype(np.uint16)
    else:
        return X.astype(np.uint32)

def apply_pca(imgs):
    x_list=[]
    for img in imgs:
        x_list.append(img.reshape((-1,3)))
    x=np.vstack(x_list)
    print(x.shape)
    pca=PCA(3).fit(x)
    x=pca.transform(x)
    x-=x.min()
    x*=255/x.max()
    
    s=0
    for i,x_aux in enumerate(x_list):
        x_list[i]=x[s:x_aux.shape[0]].reshape(imgs[i].shape)
        s+=x_aux.shape[0]

    with open('pca.pkl','wb') as f:
        pickle.dump(pca,f)

    return x_list

def apply_inverse_pca(imgs):
    with open('pca.pkl' ,'rb') as f:
        pca=pickle.load(f)

    x_list=[]
    for img in imgs:
        x_list.append(img.reshape((-1,3)))
    x=np.vstack(x_list)
    print(x.shape)
    x=pca.inverse_transform(x)

    s=0
    for i,x_aux in enumerate(x_list):
        x_list[i]=x[s:x_aux.shape[0]].reshape(imgs[i].shape)
        s+=x_aux.shape[0]

    return x_list
        


def read_dataset(name,istrain=False):
    if name =='mnist':
        X={}
        (X['train'],_), (X['test'],_) = tf.keras.datasets.mnist.load_data()
        for k in X.keys():
            X[k]=X[k].reshape(-1,28,28,1)
            #X[k]=resample(X[k],n_samples=1000,random_state=0)
            print(X[k].shape)
    
    else:
        dataset_path=name
        imgs=[]
        filenames=[]
        mean_aspect_ratio=0
        min_width=np.inf
        for w in os.listdir(dataset_path): #SOLO SE ESTA COGIENDO UN SUBSET para test!!!
            if w.endswith('.png') or w.endswith('.jpg')or w.endswith('.jpeg') or w.endswith('.gif') or w.endswith('.pgm') or w.endswith('.ppm'):       
                aux_im=Image.open(dataset_path+'/'+w)
                aspect_ratio=aux_im.size
                min_width=min(min_width,aspect_ratio[0])
                mean_aspect_ratio+=aspect_ratio[1]/aspect_ratio[0]
                imgs.append(aux_im)
                filenames.append('.'.join(w.split('.')[:-1]))

        mean_aspect_ratio/=len(imgs)
        h_resize=min_width*mean_aspect_ratio
        print(min_width,h_resize,mean_aspect_ratio)

        max_val=0
        for i,img in enumerate(imgs):
            if istrain:
                imgs[i]=img.resize((1024, 1024))
            imgs[i]=np.array(imgs[i])
            max_img=imgs[i].max()
            if max_img>max_val:
                max_val=max_img            
        
        imgs=np.array(imgs)
        
        if len(imgs.shape)==1:
            for i,img in enumerate(imgs):
                if len(img.shape)==2:
                    imgs[i]=img.reshape((1,)+img.shape+(1,))
                else:
                    imgs[i]=img.reshape((1,)+img.shape)

                imgs[i]=code_array(imgs[i],max_val)

                
        elif len(imgs.shape)==3:
            imgs=imgs#*255/65535 #CUIDADO! CONVIRTIENDO A 1B/PIXEL. Desactivar para test
            imgs=code_array(imgs.reshape(imgs.shape+(1,)),max_val)
            

        #print(imgs.min(),imgs.max())
        if istrain:
            X={}
            train_test_limit=int(imgs.shape[0]*0.75)
            X['train']=imgs[:train_test_limit]
            X['test']=imgs[train_test_limit:]
            aux_train_list=[]
            aux_test_list=[]
            aux_chan=X['train'].shape[-1]
            if aux_chan>1:
                for i in range(aux_chan):
                    aux_train_list.append(X['train'][:,:,:,i:i+1])
                    aux_test_list.append(X['test'][:,:,:,i:i+1])
                    
                X['train']=np.concatenate(aux_train_list,axis=0)
                X['test']=np.concatenate(aux_test_list,axis=0)
            #X['train'],X['test']=train_test_split(imgs,test_size=0.25,random_state=0)
            print(X['train'].shape)
            print(filenames[:train_test_limit])
            print(X['test'].shape)
            print(filenames[train_test_limit:])
        else:
            X=imgs


    return X, filenames