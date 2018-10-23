import tensorflow as tf
import sys
sys.path.append('../src')
from utils import read_dataset


tf.reset_default_graph()

img1 = tf.placeholder(tf.uint8, shape=(None,None,None,None),name='x1')
img2 = tf.placeholder(tf.uint8, shape=(None,None,None,None),name='x2')

res=tf.image.ssim(img1,img2,max_val=255)



_,dataset_path_1,dataset_path_2=sys.argv
X1,filenames1=read_dataset(dataset_path_1)
X2,filenames2=read_dataset(dataset_path_2)

with tf.Session() as sess:
    avg=0
    count=0
    for i,x1 in enumerate(X1):
        for j,x2 in enumerate(X2):
            if filenames1[i]==filenames2[j]:
                ssim=sess.run(res,feed_dict={img1:x1,img2:x2})
                avg+=ssim
                count+=1
                print('SSIM of {0}: {1}'.format(filenames1[i],ssim))
    avg/=count
    print('average SSIM: {}'.format(avg))

    
