import sys
sys.path.append('../src')
from encoder import encoder_CAE
from decoder import decoder_CAE
from utils import read_dataset
import os
import numpy as np
from PIL import Image
import json

def compress_images(imgs,cae,chanels):
    one_img=[]
    for i in range(chanels):
        one_img.append(cae.compress(imgs[:,:,:,i:i+1]))
    one_img=np.stack(one_img,axis=-1)
    return one_img

def uncompress_images(imgs,cae,chanels):
    one_img=[]
    for i in range(chanels):
        one_img.append(cae.uncompress(imgs[:,:,:,:,i]))
    if chanels==1:
        return one_img[0]
    one_img=np.stack(one_img,axis=-1)
    return one_img

def save_compressed(img,output_dir,filename):
    assert (np.round(img)-img).sum()==0
    Image.fromarray(img).save(output_dir+'/'+filename+'.png')

def save_imag(img,output_dir,filename,shape):
    Image.fromarray(img).resize(reversed(shape)).save(output_dir+'/'+filename+'.png')

    
if __name__=="__main__":
    dataset_path=sys.argv[1]
    flag=sys.argv[2]
    
    X,filenames=read_dataset(dataset_path)
    dataset_name=dataset_path.split('/')[-1].replace('_compress','')

    if len(sys.argv)==4:
        checkpoint_name=sys.argv[3].split('/')[-1]
    else:
        checkpoint_name=dataset_name

    chanels=X[0].shape[-1]

    output_dir=dataset_path.replace('_compress','')+'_'+flag
    
    try:
        os.mkdir(output_dir)
    except:
        pass

    original_shapes={}

    if flag=='compress':
        cae=encoder_CAE('checkpoints/'+checkpoint_name+'/')
    elif flag=='uncompress':
        cae=decoder_CAE('checkpoints/'+checkpoint_name+'/')
        shapes=[]
        with open('original_shape.json','r') as f:
            shapes=json.load(f)
    
    if len(X.shape)==1:
        if flag=='compress':
            for i,img in enumerate(X):
                output_img=compress_images(img,cae,chanels)
                _,aux_h,aux_w,aux_z,aux_ch=output_img.shape
                output_img=np.squeeze(output_img.reshape((aux_h*4,aux_w*aux_z//4,aux_ch)))
                save_compressed(output_img,output_dir,filenames[i]) 
                original_shapes[filenames[i]]=img.shape[1:3]
                with open('original_shape.json','w') as f:
                    json.dump(original_shapes,f)
                print(output_img[0].shape)

        elif flag=='uncompress':
            for i,img in enumerate(X):
                _,aux_h,aux_w,aux_ch=img.shape
                aux_z=32
                output_img=uncompress_images(img.reshape((1,aux_h//4,4*aux_w//aux_z,aux_z,aux_ch)),cae,chanels)
                output_img=np.squeeze(output_img)
                save_imag(output_img,output_dir,filenames[i],shapes[filenames[i]])
                print(shapes[filenames[i]])

    else:
        if flag=='compress':
            output_img=compress_images(X,cae,chanels)
            aux_n,aux_h,aux_w,aux_z,aux_ch=output_img.shape
            for i in range(aux_n):
                save_compressed(np.squeeze(output_img[i].reshape((aux_h*4,aux_w*aux_z//4,aux_ch))),output_dir,filenames[i])
            with open('original_shape.json','w') as f:
                json.dump(X.shape[1:3],f)
            print(output_img[0].shape)

        elif flag=='uncompress':
            aux_n,aux_h,aux_w,aux_ch=img.shape
            aux_z=32
            output_img=uncompress_images(X.reshape((aux_n,aux_h//4,4*aux_w//aux_z,aux_z,aux_ch)),cae,chanels)
            
            for i in range(output_img.shape[0]):
                save_imag(np.squeeze(output_img[i]),output_dir,filenames[i],shapes)
            print(shapes)

    cae.close_session()