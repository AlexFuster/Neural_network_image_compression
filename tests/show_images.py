import sys
sys.path.append('../src')
from utils import read_dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
def show_images(imlist,nrows,ncols,titles=None):
    fig=plt.figure(figsize=(3*ncols,2*nrows))
    for i in range(nrows):
        for j in range(ncols):
            index=ncols*i+j
            plt.subplot(nrows,ncols,index+1)
            plt.imshow(np.squeeze(imlist[index]))
            if titles is not None:
                plt.title(titles[index]), 
            plt.xticks([])
            plt.yticks([])
    plt.subplots_adjust(top=0.9,bottom=0.05,left=0.05,right=0.95,wspace=0.05,hspace=0.05)
    plt.show()

if __name__=="__main__":
    
    dataset_path=sys.argv[1]
    X,filenames=read_dataset(dataset_path)
    if len(X[0].shape)==4:
        show_images(X,2,2,filenames)

    elif len(X[0].shape)==5:
        _,h,w,z,chanels=X[0].shape
        aux_list=[]
        for row in X:
            _,h,w,z,chanels=row.shape
            aux_list.append(row.reshape(h*4,w*z//4,chanels))
        
        show_images(aux_list,2,2,filenames)
