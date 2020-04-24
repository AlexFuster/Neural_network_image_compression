import numpy as np
import tensorflow as tf
import sys
from utils import read_dataset

def get_ms_ssim(dataset_path_1,dataset_path_2):
    X1,filenames1=read_dataset(dataset_path_1)
    X2,filenames2=read_dataset(dataset_path_2)
    res={}
    with tf.device('/CPU:0'):
        
        count=0
        for i,x1 in enumerate(X1):
            for j,x2 in enumerate(X2):
                if filenames1[i]==filenames2[j]:
                    ssim=tf.image.ssim_multiscale(np.squeeze(x1),np.squeeze(x2),max_val=255)
                    
                    count+=1
                    res[filenames1[i]]=ssim

    return res

if __name__=='main':
    _,dataset_path_1,dataset_path_2=sys.argv
    avg=0
    res=get_ms_ssim(dataset_path_1,dataset_path_2)
    for k, v in res.items():
        print('SSIM of {0}: {1}'.format(k,v))
        avg+=v

    avg/=len(res)
    print('average SSIM: {}'.format(avg))
