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

    x,filenames=read_dataset(dataset_path)
    dataset_name=dataset_path.split('/')[-1]

    if len(sys.argv)==3:
        checkpoint_name=sys.argv[2].split('/')[-1]
    else:
        checkpoint_name=dataset_name

    #rellenar