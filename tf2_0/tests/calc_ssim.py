import numpy as np
import tensorflow as tf
import sys
sys.path.append('../src')
from utils import read_dataset

_,dataset_path_1,dataset_path_2=sys.argv
X1,filenames1=read_dataset(dataset_path_1)
X2,filenames2=read_dataset(dataset_path_2)
with tf.device('/CPU:0'):
    avg=0
    count=0
    for i,x1 in enumerate(X1):
        for j,x2 in enumerate(X2):
            if filenames1[i]==filenames2[j]:
                ssim=tf.image.ssim_multiscale(np.squeeze(x1),np.squeeze(x2),max_val=255)
                avg+=ssim
                count+=1
                print('SSIM of {0}: {1}'.format(filenames1[i],ssim))
    avg/=count
    print('average SSIM: {}'.format(avg))
