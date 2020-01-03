from utils import ProClass

def save_compressed(img,output_dir,filename):
    assert (np.round(img)-img).sum()==0
    Image.fromarray(img).save(output_dir+'/'+filename+'.png',optimize=True)

class BaseEncoder(tf.keras.Model):
    def __init__(self):
        super(BaseEncoder, self).__init__()
        self.conv1 = Conv2D(32, 5, 2, 'SAME', activation=tf.nn.leaky_relu)
        self.conv2 = Conv2D(64, 5, 2, 'SAME', activation=tf.nn.leaky_relu)
        self.conv3 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        self.conv4 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        #self.conv5 = Conv2D(64, 5, 2, 'SAME', activation=tf.nn.leaky_relu)
        #self.conv6 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        #self.conv7 = Conv2D(64, 3, 1, 'SAME', activation=tf.nn.leaky_relu)
        self.conv8 = Conv2D(32, 5, 2, 'SAME', activation=tf.nn.leaky_relu)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        res = x
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + res
        #x = self.conv5(x)
        #res = x
        #x = self.conv6(x)
        #x = self.conv7(x)
        #x = x + res
        x = self.conv8(x)
        return tf.clip_by_value(x, 0, 1)

class Encoder(ProClass):
    def __init__(self):
        super(Encoder, self).__init__(BaseEncoder)

    def __call__(self, x):
        img_norm = x.astype(np.float32) / 255

        img_channels = convert_to_colourspace(ycbcr_kernel, ycbcr_off, img_norm)

        encoded = self.run_model(img_channels)

        encoded = tf.concat(encoded, axis=3)

        return np.round(encoded * 255).astype(np.uint8)

    def _compress_batch(self,x):
        if len(x.shape==5):
            x=x[0]
        output_img=self(x)
        n,h,w,c=output_img.shape
        output_img=output_img.reshape((n,h*4,w*8,c//32)) #posibly wrong
        for i in range(aux_n):
            save_compressed(np.squeeze(output_img[i]),output_dir,filenames[i])
            print('save {}'.format(filenames[i]))

    def compress(self,dataset_path,checkpoint_path='checkpoints/encoder'):
        output_dir=dataset_path+'_compressed'

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        self.load(checkpoint_path)

        x,filenames=read_dataset(dataset_path)
        tot_n=x.shape[0]
        if len(x.shape)==1:
            batch_size=1
        else:
            batch_size=4

        for b in range(tot_n//batch_size + int((tot_n%batch_size)!=0)):
            lower_index=b*batch_size
            upper_index=min(tot_n,lower_index+batch_size)
            output_img=self._compress_batch(x[lower_index:upper_index],filenames[lower_index:upper_index])