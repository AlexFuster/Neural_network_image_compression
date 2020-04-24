import numpy as np
import tensorflow as tf
import os
from PIL import Image

ycbcr_kernel=np.array([[0.299,0.587,0.114],[-0.16874,-0.33126,0.5],[0.5,-0.41869,-0.08131]])
ycbcr_inv_kernel=np.linalg.inv(ycbcr_kernel)
ycbcr_off=np.array([0,0.5,0.5])

pca_kernel=np.array([[1/3,1/3,1/3],[-0.5,0,0.5],[0.25,-0.5,0.25]])
pca_inv_kernel=np.linalg.inv(pca_kernel)
pca_off=np.array([0,0.5,0.5])

class Model(tf.keras.Model):
    def __init__(self,model_class):
        super(Model, self).__init__()
        self.model_y=model_class()
        self.model_cbcr=model_class()
        self.train_mode=False

    def run_model(self,x):
        x_0=x[0]
        x_1=tf.concat(x[1:],axis=0)
        out_0=self.model_y(x_0)
        out_1=self.model_cbcr(x_1)
        out_1,out_2=tf.split(out_1,2,axis=0)
        return [out_0,out_1,out_2]

    def load(self,path):
        self.load_weights(path)

    def _feed_batch(self,x,filenames,output_dir,in_cshape):
        if len(x.shape)==5:
            x=x[0]
        output_img=x
        n,h,w,c=output_img.shape
        #if in_cshape==96 and c==3:
        #    output_img=np.concatenate([tf.one_hot(output_img[:,:,:,i],256,axis=-1) for i in range(3)],axis=3)

        output_img=self(tf.convert_to_tensor(output_img))
        output_img=output_img.numpy().round().astype(np.uint8)
        n,h,w,c=output_img.shape
        res={}
        for i in range(n):
            save_img(np.squeeze(output_img[i]),output_dir,filenames[i])
            res[filenames[i]]=output_img[i]
            print('save {}'.format(filenames[i]))
        return res

    def _use_model(self,dataset_path,checkpoint_path,output_dir,in_cshape):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        self.load(checkpoint_path)

        x,filenames=read_dataset(dataset_path)
        tot_n=x.shape[0]
        if len(x.shape)==1:
            batch_size=1
        else:
            batch_size=4
        res={}
        for b in range(tot_n//batch_size + int((tot_n%batch_size)!=0)):
            lower_index=b*batch_size
            upper_index=min(tot_n,lower_index+batch_size)
            res.update(self._feed_batch(x[lower_index:upper_index],filenames[lower_index:upper_index],output_dir,in_cshape))
        return res

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

def get_next_log_name(log_dir,lastdir=False):
    try:
        return log_dir+str(max([int(i.split('_')[0]) for i in os.listdir(log_dir)])+1-int(lastdir))
    except ValueError:
        return log_dir+'1'

def save_img(img,output_dir,filename):
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