import sys
sys.path.append('../src')
from encoder import encoder_CAE
from utils import read_dataset, save_compressed
import os
import numpy as np
from PIL import Image
import json

def compress_images(imgs,cae,chanels):
    return cae.compress(imgs)
    
if __name__=="__main__":
    dataset_path=sys.argv[1]
    
    X,filenames=read_dataset(dataset_path)
    dataset_name=dataset_path.split('/')[-1]

    if len(sys.argv)==3:
        checkpoint_name=sys.argv[2].split('/')[-1]
    else:
        checkpoint_name=dataset_name

    chanels=X[0].shape[-1]

    output_dir=dataset_path+'_compressed'
    
    try:
        os.mkdir(output_dir)
    except:
        pass
    
    cae=encoder_CAE('checkpoints/'+checkpoint_name+'/')
    
    if len(X.shape)==1:
        for i,img in enumerate(X):
            output_img=cae.compress(img)
            _,aux_h,aux_w,aux_z=output_img.shape
            print(output_img.shape)
            output_img=np.squeeze(output_img)
            save_compressed(output_img,output_dir,filenames[i]) 
            print(output_img.shape)

    else:
        for b in range(X.shape[0]//4 + int((X.shape[0]%8)!=0)):
            lower_index=b*4
            upper_index=min(X.shape[0],(b+1)*4)

            output_img=cae.compress(X[lower_index:upper_index])
            aux_n,aux_h,aux_w,aux_ch=output_img.shape
            for i in range(aux_n):
                save_compressed(np.squeeze(output_img[i]),output_dir,filenames[lower_index+i])
                print('save {}'.format(filenames[lower_index+i]))
            
        with open('original_shape.json','w') as f:
            json.dump(X.shape[1:3],f)
        print(output_img[0].shape)

    cae.close_session()