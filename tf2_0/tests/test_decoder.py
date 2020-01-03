import sys
sys.path.append('../src')
from decoder import decoder_CAE
from utils import read_dataset, save_imag
import os
import numpy as np
from PIL import Image
import json
    
if __name__=="__main__":
    dataset_path=sys.argv[1]
    
    X,filenames=read_dataset(dataset_path)
    dataset_name=dataset_path.split('/')[-1].replace('_compressed','')

    if len(sys.argv)==3:
        checkpoint_name=sys.argv[2].split('/')[-1]
    else:
        checkpoint_name=dataset_name

    chanels=X[0].shape[-1]

    output_dir=dataset_path.replace('_compressed','')+'_uncompressed'
    
    try:
        os.mkdir(output_dir)
    except:
        pass

    original_shapes={}

    cae=decoder_CAE('checkpoints/'+checkpoint_name+'/')
    
    if len(X.shape)==1:
        for i,img in enumerate(X):
            _,aux_h,aux_w,aux_ch=img.shape
            aux_z=32
            output_img=cae.uncompress(img)
            output_img=np.squeeze(output_img)
            save_imag(output_img,output_dir,filenames[i])

    else:
        _,aux_h,aux_w,aux_ch=X.shape
        aux_z=32
        for b in range(X.shape[0]//2 + int((X.shape[0]%2)!=0)):
            lower_index=b*2
            upper_index=min(X.shape[0],(b+1)*2)
            n_imgs=upper_index-lower_index
            output_img=cae.uncompress(X[lower_index:upper_index])
            
            for i in range(output_img.shape[0]):
                save_imag(np.squeeze(output_img[i]),output_dir,filenames[lower_index+i])
                print('save {}'.format(filenames[lower_index+i]))

    cae.close_session()