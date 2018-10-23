import numpy as np
#import pandas as pd
import tensorflow as tf
import math
from utils import encode, decode, downsample_entropy, dense_entropy

def get_png_size(xin):
    return tf.strings.length(tf.image.encode_png(xin))

class training_CAE():
    def __init__(self,log_dir,checkpoint_dir,last_epoch_file):
        self.log_dir=log_dir
        self.checkpoint_dir=checkpoint_dir
        self.last_epoch_file=last_epoch_file
    
    def train(self,X,X_val,max_epochs=10,batch_size=32,from_checkpoint=False,config='mnist'):
        _,h,w,_=X.shape
        max_val=X.max()

        tf.reset_default_graph()
        global_step = tf.Variable(0, trainable=False)
        
        logof2=tf.log(tf.constant(2,dtype=tf.float32))
        img = tf.placeholder(tf.float32, shape=(None, h, w, 1), name='x')
        entropy_loss_coef=tf.placeholder(tf.float32)
        img_norm=img/max_val
        with tf.variable_scope("encoder"):
            encoded=encode(img_norm,config)

        uniform_noise=tf.random_uniform(tf.shape(encoded),-0.5,0.5)
        noisy_encoded=encoded+uniform_noise/max_val

        #Discrete entropy
        denorm_encoded=noisy_encoded*max_val
        denorm_encoded=tf.reshape(denorm_encoded,(tf.shape(denorm_encoded)[0],-1))

        disc_probs_list=[tf.reduce_sum(tf.cast(tf.equal(tf.round(denorm_encoded),i),tf.float32),axis=1) for i in range(max_val+1)]
        disc_probs=tf.stack(disc_probs_list)/tf.cast(tf.shape(denorm_encoded)[1],tf.float32)
        disc_entropy=tf.reduce_sum(tf.multiply(disc_probs,tf.nn.relu(-tf.log(disc_probs+1e-5)/logof2)),axis=0)
        disc_entropy=tf.reshape(disc_entropy,(-1,1))

        transform_shape=tf.convert_to_tensor([1,4,8,1/32.0])
        img_to_convert=tf.reshape(tf.cast(tf.round(noisy_encoded*max_val),tf.uint8),tf.cast(tf.cast(tf.shape(noisy_encoded),tf.float32)*transform_shape,tf.int32))
        size_list=tf.cast(tf.stack(tf.map_fn(get_png_size,img_to_convert,dtype=tf.int32)),tf.float32)
        mean_size=tf.reduce_mean(size_list)
        tf.summary.scalar("mean_expected_size",mean_size)
        #Continous entropy estimation
        with tf.variable_scope("entropy"):
            if config=='mnist':
                layer=noisy_encoded
            else:
                layer=downsample_entropy(noisy_encoded)
                
            aprox_entropy=dense_entropy(layer)

        entropy_var_list=tf.global_variables(scope='entropy')

        #Classification error
        #layer=tf.layers.dense(denorm_encoded/max_val,1024,activation=tf.nn.relu)
        #layer=tf.layers.dense(layer,1024,activation=tf.nn.relu)
        #layer=tf.layers.dense(layer,10)
        #class_error=tf.losses.softmax_cross_entropy(labels,layer)
        #tf.summary.scalar("classification_loss",class_error)


        #loss of entropy estimation
        aprox_entropy_loss=tf.losses.mean_squared_error(tf.reshape(size_list,(-1,1))/10000.0,aprox_entropy)
        tf.summary.scalar("entropy_estimation_loss",aprox_entropy_loss)

        train_aprox_entropy=tf.train.AdamOptimizer(learning_rate=1e-3).minimize(aprox_entropy_loss,var_list=entropy_var_list)

        mean_disc_entropy=tf.reduce_mean(disc_entropy)
        #entropy loss
        entropy_loss=tf.reduce_mean(aprox_entropy)
        tf.summary.scalar("entropy_loss",mean_disc_entropy)

        with tf.variable_scope("decoder"): 
            decoded=decode(noisy_encoded,(h,w),config)

        #reconstruction loss
        ms_rec_loss = tf.losses.mean_squared_error(img_norm,decoded)
        tf.summary.scalar("reconstruction_loss",ms_rec_loss)

        #ssim
        ssim=tf.reduce_mean(tf.image.ssim(img_norm,decoded,max_val=1.0))
        tf.summary.scalar("SSIM",ssim)
        rec_loss =-ssim

        #learning rate decay
        learning_rate=1e-3
        #learning_rate = tf.train.exponential_decay(1e-3, global_step,8000, 0.75, staircase=True)
        
        #loss
        loss=rec_loss+entropy_loss_coef*entropy_loss#+0.1*class_error
        tf.summary.scalar("loss",loss)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
        
        merge_op=tf.summary.merge_all()
        
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:

            encoder_vars=[v for v in tf.global_variables(scope='encoder') if 'Adam' not in v.name]
            decoder_vars=[v for v in tf.global_variables(scope='decoder') if 'Adam' not in v.name]
            
            saver_train = tf.train.Saver()
            saver_encoder=tf.train.Saver(encoder_vars)
            saver_decoder=tf.train.Saver(decoder_vars)

            writer=tf.summary.FileWriter(self.log_dir)
            
            if from_checkpoint:
                latest=tf.train.latest_checkpoint(checkpoint_dir=self.checkpoint_dir+'training')
                saver_train.restore(sess, save_path=latest)
                with open(self.last_epoch_file,'r') as f:
                        i_0=int(f.read())+1
            else:
                sess.run(tf.global_variables_initializer())
                i_0=0
            
            num_batches=int(X.shape[0]/batch_size)
            old_mean_disc_entropy=0
            entropy_coef=0.01
            lower_entropy_limit,upper_entropy_limit=1.0,2.0
            entropy_range=upper_entropy_limit-lower_entropy_limit
            trigger_decrease_entropy_coef=False
            trigger_increase_entropy_coef=False
            for i in range(i_0,max_epochs):
                train_loss=0
                mean_entropy=0
                mean_ssim=0
                
                for b in range(num_batches):
                    low_index=batch_size*b
                    up_index=batch_size*(b+1)
                    
                    _, loss_, entropy_, ssim_, _=sess.run((train_op, loss, mean_disc_entropy, ssim, train_aprox_entropy),\
                        feed_dict={img:X[low_index:up_index,:],entropy_loss_coef:entropy_coef})
                    train_loss+=loss_
                    mean_entropy+=entropy_
                    mean_ssim+=ssim_

                train_loss/=num_batches
                mean_entropy/=num_batches
                mean_ssim/=num_batches
                msg='Epoch: {0}\tTraining loss: {1:.6f}\tEntropy: {2:.6f}\t SSIM: {3:.6f}'.format(i,train_loss,mean_entropy,mean_ssim)
                if X_val is not None:
                    val_loss=sess.run(loss,feed_dict={img:X_val,entropy_loss_coef:entropy_coef})
                    msg+='\tVal loss: {:.6f}'.format(val_loss)
                print(msg)
                summaries=sess.run(merge_op,feed_dict={img:X,entropy_loss_coef:entropy_coef})
                writer.add_summary(summaries,global_step=i)
                if i%10==0:
                    saver_train.save(sess, self.checkpoint_dir+'training/training.ckpt',global_step=1)
                    saver_encoder.save(sess, self.checkpoint_dir+'encoder/encoder.ckpt',global_step=1)
                    saver_decoder.save(sess, self.checkpoint_dir+'decoder/decoder.ckpt',global_step=1)
                    with open(self.last_epoch_file,'w') as f:
                        f.write(str(i))
                    print('Checkpoint saved successfully')

                
                dif_disc_entropy=mean_entropy-old_mean_disc_entropy

                if mean_entropy<lower_entropy_limit and (dif_disc_entropy<0 or abs(dif_disc_entropy)<1e-3):
                    #if trigger_decrease_entropy_coef:
                        #entropy_coef=max(1e-4,entropy_coef*(1-(0.05*(lower_entropy_limit-mean_entropy)/entropy_range)))
                    trigger_decrease_entropy_coef=not trigger_decrease_entropy_coef    
                    trigger_increase_entropy_coef=False
                elif mean_entropy>upper_entropy_limit and (dif_disc_entropy>0 or abs(dif_disc_entropy)<1e-3):
                    #if trigger_increase_entropy_coef:
                        #entropy_coef=min(0.1,entropy_coef*(1+(0.05*(mean_entropy-upper_entropy_limit)/entropy_range)))
                    trigger_increase_entropy_coef=not trigger_increase_entropy_coef
                    trigger_decrease_entropy_coef=False
                else:
                    trigger_decrease_entropy_coef=False
                    trigger_increase_entropy_coef=False

                print('Entropy coef: {0}, {1}'.format(entropy_coef,trigger_increase_entropy_coef))

                old_mean_disc_entropy=mean_entropy
                    
            return self            
        