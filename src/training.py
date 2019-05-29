import numpy as np
import os
import tensorflow as tf
import math
from utils import encode, decode, downsample_entropy, dense_entropy
from utils import convert_to_rgb, convert_to_colourspace,ycbcr_kernel,ycbcr_inv_kernel,ycbcr_off,pca_kernel,pca_inv_kernel,pca_off
import matplotlib.pyplot as plt
from utils import save_imag

from PIL import Image
get_png_size=lambda xin: tf.strings.length(tf.image.encode_png(xin))

class training_CAE():
    def __init__(self,log_dir,checkpoint_dir,last_epoch_file):
        self.log_dir=log_dir
        self.checkpoint_dir=checkpoint_dir
        self.last_epoch_file=last_epoch_file
    
    def train(self,X,X_val,max_epochs,batch_size,from_checkpoint=False):
        _,h,w,_=X.shape
        if X.dtype==np.uint8:
            max_val=255
        
        assert max_val==255
        tf.reset_default_graph()
        global_step = tf.Variable(0, trainable=False)
        
        logof2=tf.log(tf.constant(2,dtype=tf.float32))
        img = tf.placeholder(tf.float32, shape=(None, h, w, 3), name='x')

        img_test = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='x_test')

        entropy_loss_coef=tf.placeholder(tf.float32)
        img_norm=img/max_val
        img_norm_test=img_test/max_val

        img_norm=tf.image.random_flip_left_right(img_norm)
        img_norm=tf.image.random_flip_up_down(img_norm)

        img_0,img_1,img_2=convert_to_colourspace(ycbcr_kernel,ycbcr_off,img_norm)
        img_test_0,img_test_1,img_test_2=convert_to_colourspace(ycbcr_kernel,ycbcr_off,img_norm_test)

        with tf.variable_scope("encoder",reuse=tf.AUTO_REUSE):
            encoded_0=tf.clip_by_value(encode(img_0,0),0,1)
            encoded_1=tf.clip_by_value(encode(img_1,1),0,1)
            encoded_2=tf.clip_by_value(encode(img_2,1),0,1)

            encoded_test_0=tf.round(tf.clip_by_value(encode(img_test_0,0),0,1)*max_val)/max_val
            encoded_test_1=tf.round(tf.clip_by_value(encode(img_test_1,1),0,1)*max_val)/max_val
            encoded_test_2=tf.round(tf.clip_by_value(encode(img_test_2,1),0,1)*max_val)/max_val
        
        #Discrete entropy

        encoded=tf.concat([encoded_0,encoded_1,encoded_2],axis=3)

        uniform_noise=tf.random_uniform(tf.shape(encoded),-0.5,0.5)/max_val
        noise_0,noise_1,noise_2=tf.split(uniform_noise,3,axis=3)
        noisy_encoded_0=tf.clip_by_value(encoded_0+noise_0,0,1)
        noisy_encoded_1=tf.clip_by_value(encoded_1+noise_1,0,1)
        noisy_encoded_2=tf.clip_by_value(encoded_2+noise_2,0,1)

        batch_encoded=tf.concat([encoded_0,encoded_1,encoded_2],axis=0)
        encoded_denorm=batch_encoded*max_val
        
        
        encoded_denorm_flat=tf.reshape(encoded_denorm,(tf.shape(encoded_denorm)[0],-1))

        disc_probs_list=[tf.reduce_sum(tf.cast(tf.equal(tf.round(encoded_denorm_flat),i),tf.float32),axis=1) for i in range(max_val+1)]
        disc_probs=tf.stack(disc_probs_list)/tf.cast(tf.shape(encoded_denorm_flat)[1],tf.float32)
        disc_entropy=tf.reduce_sum(tf.multiply(disc_probs,-tf.log(tf.clip_by_value(disc_probs,1e-5,1.0))/logof2),axis=0)
        disc_entropy=tf.reshape(disc_entropy,(-1,1))

        transform_shape=tf.convert_to_tensor([1,4,8,1/32.0])
        img_to_convert=tf.reshape(tf.cast(tf.round(encoded_denorm),tf.uint8),tf.cast(tf.cast(tf.shape(encoded_denorm),tf.float32)*transform_shape,tf.int32))
        
        size_list=tf.cast(tf.stack(tf.map_fn(get_png_size,img_to_convert,dtype=tf.int32)),tf.float32)
        mean_size=tf.reduce_mean(size_list)
        tf.summary.scalar("mean_expected_size",mean_size)
        #Continous entropy estimation
        with tf.variable_scope("entropy",reuse=tf.AUTO_REUSE):
            layer=downsample_entropy(batch_encoded)

            #aprox_entropy=tf.clip_by_value(dense_entropy(layer),0,8)
            aprox_entropy=dense_entropy(layer)

        aprox_entropy_0,aprox_entropy_1,aprox_entropy_2=tf.split(aprox_entropy,3,axis=0)

        entropy_var_list=tf.global_variables(scope='entropy')


        #loss of entropy estimation
        #aprox_entropy_loss=tf.losses.mean_squared_error(disc_entropy,aprox_entropy)
        aprox_entropy_loss=tf.losses.mean_squared_error(tf.reshape(size_list,(-1,1))/1000.0,aprox_entropy)
        tf.summary.scalar("entropy_estimation_loss",aprox_entropy_loss)

        train_aprox_entropy=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(aprox_entropy_loss,var_list=entropy_var_list)

        mean_disc_entropy=tf.reduce_mean(disc_entropy)
        #entropy loss
        entropy_loss_0=tf.reduce_mean(aprox_entropy_0)
        entropy_loss_1=tf.reduce_mean(aprox_entropy_1)
        entropy_loss_2=tf.reduce_mean(aprox_entropy_2)
        tf.summary.scalar("entropy_loss",mean_disc_entropy)


        with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE): 
            decoded_0=tf.clip_by_value(decode(noisy_encoded_0,0),0,1)
            decoded_1=tf.clip_by_value(decode(noisy_encoded_1,1),0,1)
            decoded_2=tf.clip_by_value(decode(noisy_encoded_2,1),0,1)

            decoded_test_0=tf.clip_by_value(decode(encoded_test_0,0),0,1)
            decoded_test_1=tf.clip_by_value(decode(encoded_test_1,1),0,1)
            decoded_test_2=tf.clip_by_value(decode(encoded_test_2,1),0,1)


        decoded=tf.concat([decoded_0,decoded_1,decoded_2],axis=3)
        
        decoded_rgb=tf.clip_by_value(convert_to_rgb(ycbcr_inv_kernel,ycbcr_off,decoded_0,decoded_1,decoded_2),0,1)
        decoded_rgb_test=tf.clip_by_value(convert_to_rgb(ycbcr_inv_kernel,ycbcr_off,decoded_test_0,decoded_test_1,decoded_test_2),0,1)

        decoded_rgb_denorm_test=tf.round(decoded_rgb_test*max_val)


        #ssim
        ssim_0=tf.reduce_mean(tf.image.ssim_multiscale(img_0,decoded_0,max_val=1.0))
        ssim_1=tf.reduce_mean(tf.image.ssim_multiscale(img_1,decoded_1,max_val=1.0))
        ssim_2=tf.reduce_mean(tf.image.ssim_multiscale(img_2,decoded_2,max_val=1.0))

        ssim_rgb=tf.reduce_mean(tf.image.ssim_multiscale(img_norm,decoded_rgb,max_val=1.0))
        ssim=(ssim_0+ssim_1+ssim_2)/3
        tf.summary.scalar("SSIM",ssim)
        rec_loss_0 =(1-ssim_0)/2
        rec_loss_1 =(1-ssim_1)/2
        rec_loss_2 =(1-ssim_2)/2

        #learning rate decay
        learning_rate=1e-4
        #learning_rate = tf.train.exponential_decay(1e-3, global_step,1500, 0.75, staircase=True)
        
        #loss
        loss_0=rec_loss_0+entropy_loss_coef*entropy_loss_0
        loss_1=(rec_loss_1+rec_loss_2+entropy_loss_coef*entropy_loss_1+entropy_loss_coef*entropy_loss_2)/2
        loss=(loss_0+2*loss_1)/3

        tf.summary.scalar("loss",loss)
        train_op_0 = tf.train.AdamOptimizer(learning_rate).minimize(loss_0,global_step=global_step)
        train_op_1 = tf.train.AdamOptimizer(learning_rate).minimize(loss_1,global_step=global_step)
        
        merge_op=tf.summary.merge_all()

        print([v.name for v in tf.global_variables()])
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            encoder_vars=[v for v in tf.global_variables(scope='encoder') if 'Adam' not in v.name]
            decoder_vars=[v for v in tf.global_variables(scope='decoder') if 'Adam' not in v.name]
            
            saver_train = tf.train.Saver()
            saver_encoder=tf.train.Saver(encoder_vars)
            saver_decoder=tf.train.Saver(decoder_vars)

            writer=tf.summary.FileWriter(self.log_dir,sess.graph)
            
            if from_checkpoint:
                latest=tf.train.latest_checkpoint(checkpoint_dir=self.checkpoint_dir+'training')
                saver_train.restore(sess, save_path=latest)
                with open(self.last_epoch_file,'r') as f:
                        i_0=int(f.read())+1
            else:
                sess.run(tf.global_variables_initializer())
                i_0=0
            
            entropy_coef=0.05#01

            num_batches=X.shape[0]//batch_size

            accepted_flag=False

            for i in range(i_0,max_epochs):
                train_loss=0
                mean_entropy=0
                mean_ssim=0
                mean_ssim_rgb=0
                np.random.shuffle(X)
                for j in range(num_batches):
                    lower_index=batch_size*j
                    upper_index=lower_index+batch_size
                    feed_dict={img:X[lower_index:upper_index],entropy_loss_coef:entropy_coef}
                    _, _, loss_, entropy_, ssim_, ssim_rgb_, _, summaries=sess.run((
                        train_op_0,
                        train_op_1,
                        loss,
                        mean_disc_entropy,
                        ssim,
                        ssim_rgb,
                        train_aprox_entropy,
                        merge_op
                        ),feed_dict=feed_dict)
                    train_loss+=loss_
                    mean_entropy+=entropy_
                    mean_ssim+=ssim_
                    mean_ssim_rgb+=ssim_rgb_
                    writer.add_summary(summaries,global_step=i*num_batches+j)
                        
                train_loss/=num_batches
                mean_entropy/=num_batches
                mean_ssim/=num_batches
                mean_ssim_rgb/=num_batches
                msg='Epoch: {0}\tTraining loss: {1:.6f}\tEntropy: {2:.6f}\tSSIM: {3:.6f}\tSSIM_rgb: {4:.6f}'.format(i,train_loss,mean_entropy,mean_ssim,mean_ssim_rgb)
                print(msg)

                if mean_ssim_rgb>0.6 and mean_entropy<=0.6:
                    accepted_flag=True
                if (not accepted_flag or mean_ssim_rgb>0.6) and mean_entropy<=0.75:

                    decoded_=sess.run(decoded_rgb_denorm_test,feed_dict={img_test:X_val})
                    rec_img=np.round(decoded_).astype(np.uint8)
                    for j,rec_img_ in enumerate(rec_img):
                        save_imag(rec_img_,'../data/traain_feedback','an_image_unc_'+str(j))

                    saver_train.save(sess, self.checkpoint_dir+'training/training.ckpt',global_step=1)
                    saver_encoder.save(sess, self.checkpoint_dir+'encoder/encoder.ckpt',global_step=1)
                    saver_decoder.save(sess, self.checkpoint_dir+'decoder/decoder.ckpt',global_step=1)
                    with open(self.last_epoch_file,'w') as f:
                        f.write(str(i))
                    print('Checkpoint saved successfully')

            input('Press Enter to finish')
                    
            return self