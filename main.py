import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, backend, losses
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, BatchNormalization, Concatenate, Activation
from PIL import Image
import numpy as np
from skeras import plot_loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import random
import time
from scipy.special import expit

import sys
sys.path.append('D:\\hey\\jk\\Journals\\Crossing_point')
from processing.diff import difference
from processing.basic import otsuthreshold,threshold

diff = difference()

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

 
def my_func(arg):
    #w = tf.Variable(arg)
    
    npf = arg.numpy()
    sigmoid = (expit(diff.sum_shift(npf,3,3))).astype('uint8')
    otsu = otsuthreshold(sigmoid)
    x_h, x_r = otsu.shape
    x_array = np.reshape(otsu,(x_h*x_r,) )
    X_tar = ((list(x_array).count(255.0)))

    #arg1 = tf.convert_to_tensor(X_tar, dtype=tf.float32)
    return X_tar




def npcc_loss_f(y_true, y_pred):  
    x = y_true
    y = y_pred
    mx = backend.mean(x)
    my = backend.mean(y)
    xm, ym = x-mx, y-my
    
    #XX = xm.numpy()
    #YY = ym.numpy()
    
    #XX_ = (tf.placeholder(XX))
    #YY_ = (tf.placeholder(YY))
    
    xx = my_func(xm)
    yy = my_func(ym)
    
    r_num = backend.sum(tf.multiply(xx,yy))
    r_den = backend.sqrt(tf.multiply(backend.sum(backend.square(xm)), backend.sum(backend.square(ym)))) + 1e-12
        
    x1=backend.cast(x, dtype='complex64')
    y1=backend.cast(y, dtype='complex64')
    
    xf = backend.abs(tf.signal.fft2d(x1))
    yf = backend.abs(tf.signal.fft2d(y1))
    mxf = backend.mean(xf)
    myf = backend.mean(yf)
    xmf, ymf = xf-mxf, yf-myf
    r_numf = backend.sum(tf.multiply(xmf,ymf))
    r_denf = backend.sqrt(tf.multiply(backend.sum(backend.square(xmf)), backend.sum(backend.square(ymf)))) + 1e-12
    r = - ((r_numf / r_denf)/2 + (r_num / r_den)/2 )
    
    #print(r)
    return r_num

def shift_loss__(y_true, y_pred):
    #x = y_true
    #y = y_pred
    #r = []
    
    
    
    #for i in range(args.batch_size):
    x1 = y_true#x[i]
    y1 = y_pred#y[i]
    
    mx = backend.mean(x1)
    my = backend.mean(y1)
    xm, ym = x1-mx, y1-my
    
    
    
    
    
    
    
    xm_ = tf.reshape(xm,(128,128))
    ym_ = tf.reshape(ym,(128,128))
    
    
    
    XX = xm_.numpy()
    YY = ym_.numpy()
    
    
    
    
    
    
    x_ = (expit(diff.sum_shift(XX,3,3))).astype('uint8')
    y_ = (expit(diff.sum_shift(YY,3,3))).astype('uint8')
    
    
    
    x_otsu = otsuthreshold(x_)
    y_otsu = otsuthreshold(y_)
    
    x_h, x_r = x_otsu.shape
    y_h, y_r = y_otsu.shape
    
 
    
    #x_array = x_otsu.reshape(x_h*x_r,)
    #y_array = y_otsu.reshape(y_h*y_r,)
    
    x_array = np.reshape(x_otsu,(x_h*x_r,) )
    y_array = np.reshape(y_otsu,(x_h*x_r,) )
    
    
    #여기까지 numpy
    X_tar = ((list(x_array).count(255.0)))
    X_est =  ((list(y_array).count(255.0)))
    overlap2 = ((list(x_array|y_array).count(255.0)*2))
    
    
    x_tar_tf = tf.constant(X_tar)
    x_est_tf = tf.constant(X_est)
    overlap2_tf = tf.constant(overlap2)
    
   
    los = (1 - (overlap2/(X_tar+X_est+ 1e-12)))
        
        #x_array_ = tf.dtypes.cast(x_array,tf.float32)
        #y_array_ = tf.dtypes.cast(y_array,tf.float32)
        
        
        
        
        #X_tar = tf_count(x_array_,255.)
        #X_est = tf_count(y_array_,255.)
        
        #print(X_tar)
        
        #print(x)
        
        #overlap2 = tf_count((x_array_|y_array_),255.) * 2
        
        #r.append(1 - overlap2/(X_tar+X_est+ 1e-12))
    
    #r_ = np.mean(r)
    
    #los = my_func(r_)    
    #print(los) 
    
    return (los)
    

def shift_loss(y_true, y_pred):
      
    x = y_true
    y = y_pred
    
    
    

    def post_process(x,y):
        
        los_ = []
        for i in range(args.batch_size):
            
            
            XX = x[i].numpy()
            YY = y[i].numpy()

            XX_ = XX.reshape(128,128)
            YY_ = YY.reshape(128,128)
            
            X_ = np.nan_to_num(XX_)
            Y_ = np.nan_to_num(YY_)
            
            
            loss_ = mean_absolute_error(X_, Y_)
            
            #print(loss_)
    
            '''
            x_ = (((diff.sum_shift(XX_,3,3)))*255).astype('uint8') #sigmoid는 일단 제외
            y_ = (((diff.sum_shift(YY_,3,3)))*255).astype('uint8')
            
            
            
            #x_otsu = otsuthreshold(x_)
            #y_otsu = otsuthreshold(y_)
                        
            #print(y_otsu[50])
            
            
            los = mean_absolute_error(x_, y_)
            
            x_h, x_r = x_otsu.shape
            y_h, y_r = y_otsu.shape
            
            x_array = np.reshape(x_otsu,(x_h*x_r,) )
            y_array = np.reshape(y_otsu,(x_h*x_r,) )
            
            X_tar = ((list(x_array).count(255)))
            X_est =  ((list(y_array).count(255)))
            overlap2 = ((list(x_array&y_array).count(255)*2))
            
            
            
            #print(x_[50])
            #print(X_tar,X_est)
            print(X_tar,X_est,overlap2)
            
            los  =  ( -(overlap2*2/(X_tar+X_est)))
            '''
            
            los_.append(loss_)
            return np.mean(los_)
    
     
    loss_ = tf.py_function(func = post_process, inp =[x,y], Tout=tf.float32)
    #print(loss_)
    
    
    return loss_ 


def npcc_loss(y_true, y_pred):  
    x = y_true
    y = y_pred
    mx = backend.mean(x)
    my = backend.mean(y)
    xm, ym = x-mx, y-my
    r_num = backend.sum(tf.multiply(xm,ym))
    r_den = backend.sqrt(tf.multiply(backend.sum(backend.square(xm)), backend.sum(backend.square(ym)))) + 1e-12
    

    r = - (r_num / r_den)
    
    print(r)
    
    return r

class holoUNET(models.Model):
    def __init__(self, org_shape, n_ch):
        ic = 3 if backend.image_data_format() == 'channels_last' else 1

        def conv_init(x, n_f, mp_flag=True):
            x = MaxPooling2D((2, 2), padding='same')(x) if mp_flag else x
            x = Conv2D(n_f, (5, 5), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(n_f, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x
        
        def conv(x, n_f, mp_flag=True):
            x = MaxPooling2D((2, 2), padding='same')(x) if mp_flag else x
            x = Conv2D(n_f, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(n_f, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x

        def deconv_unet(x, e, n_f):
            x = UpSampling2D((2, 2))(x)
            x = Concatenate(axis=ic)([x, e])
            x = Conv2D(n_f, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(n_f, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x
        
        def deconv_unet_end(x, e, n_f):
            x = UpSampling2D((2, 2))(x)
            x = Concatenate(axis=ic)([x, e])
            x = Conv2D(n_f, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(n_f, (5, 5), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x
        
        # Input
        original = Input(shape=org_shape)
        
        # Encoding
        c1 = conv_init(original, 32, mp_flag=False)
        c2 = conv(c1, 64)
        c3 = conv(c2, 128)
        c4 = conv(c3, 256)
        c5 = conv(c4, 512)
        # Encoder
        encoded = conv(c5, 1024)
        

        # Decoding
        x = deconv_unet(encoded, c5, 512)
        x = deconv_unet(x, c4, 256)
        x = deconv_unet(x, c3, 128)
        x = deconv_unet(x, c2, 64)
        x = deconv_unet_end(x, c1, 32)

        decoded =  Conv2D(n_ch, (3, 3), activation='sigmoid', padding='same')(x)
        #decoded = Conv2D(n_ch, (3, 3), padding='same')(x)
        
        #print(shift_loss)
        #XX = decoded.numpy()
        #arg1 = tf.convert_to_tensor(XX, dtype=tf.float32)
        
        super().__init__(original, decoded)
        self.compile(optimizer='adadelta', loss=shift_loss,run_eagerly=True)
        #self.summary() 
        
class UNET(models.Model):
    def __init__(self, org_shape, n_ch):
        ic = 3 if backend.image_data_format() == 'channels_last' else 1

        def conv(x, n_f, mp_flag=True):
            x = MaxPooling2D((2, 2), padding='same')(x) if mp_flag else x
            x = Conv2D(n_f, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('tanh')(x)
            x = Dropout(0.05)(x)
            x = Conv2D(n_f, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('tanh')(x)
            return x

        def deconv_unet(x, e, n_f):
            x = UpSampling2D((2, 2))(x)
            x = Concatenate(axis=ic)([x, e])
            x = Conv2D(n_f, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('tanh')(x)
            x = Conv2D(n_f, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('tanh')(x)
            return x

        # Input
        original = Input(shape=org_shape)

        # Encoding
        c1 = conv(original, 32, mp_flag=False)
        c2 = conv(c1, 64)
        # Encoder
        encoded = conv(c2, 128)

        # Decoding
        x = deconv_unet(encoded, c2, 128)
        x = deconv_unet(x, c1, 32)

        decoded = Conv2D(n_ch, (3, 3), activation='sigmoid', padding='same')(x)
        #decoded = Conv2D(n_ch, (3, 3), padding='same')(x)
        
        super().__init__(original, decoded)
        self.compile(optimizer='adadelta', loss='npcc_loss_f')
        #self.summary() 
###########################
# Data preparation
########################### 
    
class DATA():
    def __init__(self,in_ch=None,shuffle=[]):
        minMaxScaler = MinMaxScaler()
        d=[]
        for i in range(1000):
            idd=str(shuffle[i]+1)
            
            st='./dataset_new/bf/data'+idd.zfill(4)+'.tif' 
            imt = Image.open(st)
            img_float = Image.fromarray(np.divide(np.array(imt), 2**8-1))            
            img_oct = img_float.convert('L')
            im = img_oct.resize((128,128))
            minMaxScaler.fit(np.array(im))
            tm=minMaxScaler.transform(np.array(im))
            d.append(tm)
        
        x_train=np.array(d)                      
                
        d=[]
        for i in range(99):
            
            idd=str(shuffle[i+1000]+1)
            st='./dataset_new/bf/data'+idd.zfill(4)+'.tif' 
            imt = Image.open(st)
            img_float = Image.fromarray(np.divide(np.array(imt), 2**8-1))            
            img_oct = img_float.convert('L')
            im = img_oct.resize((128,128))
            minMaxScaler.fit(np.array(im))
            tm=minMaxScaler.transform(np.array(im))
            d.append(tm)    
        
        x_test=np.array(d)  
        
        d=[]
        for i in range(1000):
            idd=str(shuffle[i])
            
            st='./dataset_new/fourier/data'+idd.zfill(4)+'.tif' 
            imt = Image.open(st)
            img_float = Image.fromarray(np.divide(np.array(imt), 2**8-1))            
            img_oct = img_float.convert('L')
            im = img_oct.resize((128,128))
            minMaxScaler.fit(np.array(im))
            tm=minMaxScaler.transform(np.array(im))
            d.append(tm)
        
        x_train_out=np.array(d)            
        
        d=[]
        for i in range(99):
            
            idd=str(shuffle[i+1000])
            st='./dataset_new/fourier/data'+idd.zfill(4)+'.tif' 
            imt = Image.open(st)
            img_float = Image.fromarray(np.divide(np.array(imt), 2**8-1))            
            img_oct = img_float.convert('L')
            im = img_oct.resize((128,128))
            minMaxScaler.fit(np.array(im))
            tm=minMaxScaler.transform(np.array(im))
            d.append(tm)
        
        x_test_out=np.array(d)
            
        if x_train.ndim == 4:
            if backend.image_data_format() == 'channels_first':
                n_ch, img_rows, img_cols = x_train.shape[1:]
            else:
                img_rows, img_cols, n_ch = x_train.shape[1:]
        else:
            img_rows, img_cols = x_train.shape[1:]
            n_ch = 1
        # in_ch can be 1 for changing BW to color image using UNet
        in_ch = n_ch if in_ch is None else in_ch

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        
        
    #    x_train /= 255
    #    x_test /= 255
        x_train_out = x_train_out.astype('float32')
        x_test_out = x_test_out.astype('float32')
   #     x_train_out /= 255
   #     x_test_out /= 255    
        
        if backend.image_data_format() == 'channels_first':
            x_train_in = x_train.reshape(x_train.shape[0], n_ch, img_rows, img_cols)
            x_test_in = x_test.reshape(x_test.shape[0], n_ch, img_rows, img_cols)
            x_train_out = x_train_out.reshape(x_train_out.shape[0], n_ch, img_rows, img_cols)
            x_test_out = x_test_out.reshape(x_test_out.shape[0], n_ch, img_rows, img_cols)
            input_shape = (in_ch, img_rows, img_cols)
        else:
            x_train_in = x_train.reshape(x_train.shape[0], img_rows, img_cols, n_ch)
            x_test_in = x_test.reshape(x_test.shape[0], img_rows, img_cols, n_ch)
            x_train_out = x_train_out.reshape(x_train_out.shape[0], img_rows, img_cols, n_ch)
            x_test_out = x_test_out.reshape(x_test_out.shape[0], img_rows, img_cols, n_ch)
            input_shape = (img_rows, img_cols, in_ch)
        
        
        #print(x_train_in.shape,x_train_out.shape)
        
        self.input_shape = input_shape
        self.x_train_in, self.x_train_out = x_train_in, x_train_out
        self.x_test_in, self.x_test_out = x_test_in, x_test_out
        self.n_ch = n_ch
        self.in_ch = in_ch
            
    
###########################
# Training and evaluation 
###########################
        
def error_calculation(data,unet):
    x_test_in = data.x_test_in
    x_test_out=data.x_test_out
    
    start = time.time()
    decoded_imgs_org = unet.predict(x_test_in)
    ts_t=time.time() - start
    print("testing time :", ts_t)
    
    decoded_imgs = decoded_imgs_org
    
    if backend.image_data_format() == 'channels_first':
        print(x_test_out.shape)
        x_test_out = x_test_out.swapaxes(1, 3).swapaxes(1, 2)
        print(x_test_out.shape)
        decoded_imgs = decoded_imgs.swapaxes(1, 3).swapaxes(1, 2)
        if data.in_ch == 1:
            x_test_in = x_test_in[:, 0, ...]
        elif data.in_ch == 2:
            print(x_test_out.shape)
            x_test_in_tmp = np.zeros_like(x_test_out)
            x_test_in = x_test_in.swapaxes(1, 3).swapaxes(1, 2)
            x_test_in_tmp[..., :2] = x_test_in
            x_test_in = x_test_in_tmp
        else:
            x_test_in = x_test_in.swapaxes(1, 3).swapaxes(1, 2)
    else:
        # x_test_in = x_test_in[..., 0]
        if data.in_ch == 1:
            x_test_in = x_test_in[..., 0]
        elif data.in_ch == 2:
            x_test_in_tmp = np.zeros_like(x_test_out)
            x_test_in_tmp[..., :2] = x_test_in
            x_test_in = x_test_in_tmp
            
    n = x_test_out.shape[0]
    plt.figure(figsize=(20, 6)) 
    
    mae=0
    mse=0
    minnn=1
    maxxx=0
    minnum=0
    maxnum=0
    for i in range(n):    
        maxx=np.squeeze(decoded_imgs[i]).max()
        minn=np.squeeze(decoded_imgs[i]).min()
        t1=((np.array(np.squeeze(decoded_imgs[i]))-minn)/(maxx-minn)).flatten()
        maxx2=np.squeeze(x_test_out[i]).max()
        minn2=np.squeeze(x_test_out[i]).min()
        t2=((np.array(np.squeeze(x_test_out[i]))-minn2)/(maxx2-minn2)).flatten()
#        minMaxScaler = MinMaxScaler()
#        minMaxScaler.fit(np.squeeze(decoded_imgs[i]))
#        t1=minMaxScaler.transform(np.squeeze(decoded_imgs[i])).flatten()
#        minMaxScaler = MinMaxScaler()
#        minMaxScaler.fit(np.squeeze(x_test_out[i]))
#        t2=minMaxScaler.transform(np.squeeze(x_test_out[i])).flatten()
        mse+=np.square(np.subtract(t1,t2)).mean()
        if np.square(np.subtract(t1,t2)).mean()>maxxx:
            maxxx=np.square(np.subtract(t1,t2)).mean()
            maxnum=i+1
        if np.square(np.subtract(t1,t2)).mean()<minnn:
            minnn=np.square(np.subtract(t1,t2)).mean()
            minnum=i+1
        print(str(np.square(np.subtract(t1,t2)).mean()))
        #print(str(i+1)+'th MSE:'+str(np.square(np.subtract(t1,t2)).mean()))
        im = Image.fromarray(255*np.squeeze(np.array(np.squeeze(decoded_imgs[i]))-minn)/(maxx-minn))
        im = im.convert("L")
        im.save("im"+str(i+1)+".png")
        mae+=np.abs(np.subtract(t1,t2)).mean()
    print(str(minnum)+','+str(minnn)+','+str(maxnum)+','+str(maxxx))
    print('mean square error: '+str(mse/n))
    print('mean absolute error: '+str(mae/n))    
    
    return mse/n,mae/n,ts_t

def show_images(data, unet):
    x_test_in = data.x_test_in
    x_test_out = data.x_test_out
    decoded_imgs_org = unet.predict(x_test_in)
    decoded_imgs = decoded_imgs_org

    if backend.image_data_format() == 'channels_first':
        print(x_test_out.shape)
        x_test_out = x_test_out.swapaxes(1, 3).swapaxes(1, 2)
        print(x_test_out.shape)
        decoded_imgs = decoded_imgs.swapaxes(1, 3).swapaxes(1, 2)
        if data.in_ch == 1:
            x_test_in = x_test_in[:, 0, ...]
        elif data.in_ch == 2:
            print(x_test_out.shape)
            x_test_in_tmp = np.zeros_like(x_test_out)
            x_test_in = x_test_in.swapaxes(1, 3).swapaxes(1, 2)
            x_test_in_tmp[..., :2] = x_test_in
            x_test_in = x_test_in_tmp
        else:
            x_test_in = x_test_in.swapaxes(1, 3).swapaxes(1, 2)
    else:
        # x_test_in = x_test_in[..., 0]
        if data.in_ch == 1:
            x_test_in = x_test_in[..., 0]
        elif data.in_ch == 2:
            x_test_in_tmp = np.zeros_like(x_test_out)
            x_test_in_tmp[..., :2] = x_test_in
            x_test_in = x_test_in_tmp
    
    
    minMaxScaler = MinMaxScaler()    
    n = 10
    plt.figure(figsize=(20, 6))
    for i in range(n):
        
        ax = plt.subplot(3, n, i + 1)
        if x_test_in.ndim < 4:
            plt.imshow(np.squeeze(x_test_in[i]), cmap='gray')
        else:
            plt.imshow(np.squeeze(x_test_in[i]), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)      
        if decoded_imgs.ndim < 4:
            minMaxScaler.fit(np.squeeze(decoded_imgs[i]))
            plt.imshow(minMaxScaler.transform(np.squeeze(decoded_imgs[i])), cmap='gray')
        else:
            minMaxScaler.fit(np.squeeze(decoded_imgs[i]))
            plt.imshow(minMaxScaler.transform(np.squeeze(decoded_imgs[i])), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n * 2)
        if x_test_out.ndim < 4:
            minMaxScaler.fit(np.squeeze(x_test_out[i]))
            plt.imshow(minMaxScaler.transform(np.squeeze(x_test_out[i])), cmap='gray')
        else:
            minMaxScaler.fit(np.squeeze(x_test_out[i]))
            plt.imshow(minMaxScaler.transform(np.squeeze(x_test_out[i])), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()



def main(in_ch=1, epochs=10, batch_size=10, shuffle=[], fig=True):

    data = DATA(in_ch=in_ch,shuffle=shuffle)
    
    start = time.time()
    unet = holoUNET(data.input_shape, data.n_ch)

    history = unet.fit(data.x_train_in, data.x_train_out,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=True,
                       validation_split=0.2)
    tr_t=time.time() - start
    print("training time :", tr_t)
    mse_t,mae_t,ts_t=error_calculation(data,unet)

    
    if fig:
        plot_loss(history)
        show_images(data, unet)
    
    return mse_t,mae_t,tr_t,ts_t            

if __name__ == '__main__':
    import argparse
    from distutils import util
    #random.seed(10)    
    mse=[]
    mae=[]
    tr=[]
    ts=[]
    for i in range(1):
        print(i)
        l=list(range(1099))
        #random.shuffle(l)

        parser = argparse.ArgumentParser(description='UNET for BRL')
        parser.add_argument('--input_channels', type=int, default=1,
                            help='input channels (default: 1)')
        parser.add_argument('--epochs', type=int, default=15,
                            help='training epochs (default: 10)')
        parser.add_argument('--batch_size', type=int, default=50,
                            help='batch size (default: 1000)')
        parser.add_argument('--fig', type=lambda x: bool(util.strtobool(x)),
                            default=True, help='flag to show figures (default: True)')
        parser.add_argument('--shuffle', type=list,
                            default=l)
        args = parser.parse_args()
    
        print("Aargs:", args)
    
        print(args.fig)
        mse_t,mae_t,tr_t,ts_t=main(args.input_channels, args.epochs, args.batch_size, args.shuffle, args.fig)
        
        mse.append(mse_t)
        mae.append(mae_t)
        tr.append(tr_t)
        ts.append(ts_t)
        