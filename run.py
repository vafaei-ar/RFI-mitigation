import matplotlib as mpl
mpl.use('agg')
import pylab as plt

import os
import argparse
import numpy as np

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Add
from tensorflow.keras.layers import Conv3D, Concatenate, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
#from keras.layers import Input, Dense, MaxPooling2D, MaxPooling3D, Dropout, BatchNormalization
from utils import *


parser = argparse.ArgumentParser()
#parser.add_argument('--ws', required=False, help='time window', type=int, default=100)
parser.add_argument('--mode', required=False, help='theme', type=int, default=1)
parser.add_argument('--w1', required=False, help='nt', type=int, default=4000)
parser.add_argument('--w2', required=False, help='nt', type=int, default=50)
parser.add_argument('--ntry', required=True, help='nt', type=str)
parser.add_argument('--batch_size', required=False, help='nt', type=int, default=0)
parser.add_argument('--epochs', required=False, help='nt', type=int, default=500)
parser.add_argument('--verbose', required=False, help='nt', type=int, default=5)
parser.add_argument('--learning_rate', required=False, help='nt',type=float, default=0.05)
parser.add_argument('--decay_rate', required=False, help='nt',type=float, default=0.985)

args = parser.parse_args()
mode = args.mode
w1 = args.w1
w2 = args.w2
n_try = args.ntry
ns = args.batch_size
training_epochs = args.epochs #Not actual epoch
verbose = args.verbose
learning_rate = args.learning_rate
decay_rate = args.decay_rate
phase_in = False
phase_out = False

if mode!=4:
    def conv(x,filters=12):
        return Conv2D(filters=filters,
                      kernel_size=(5,5),
                      strides=(1,1),
                      activation='relu',
                      padding='same',
                      kernel_initializer='glorot_normal',
#                      bias_initializer='zeros',
                      kernel_regularizer=l2(0.05),
                      bias_regularizer=l2(0.05))(x)
    window=[0,w1,w2,0]
else:
    def conv(x,filters=6):
        return Conv3D(filters=filters,
                      kernel_size=(5,5,5),
                      strides=(1, 1, 1),
                      padding='same',
                      data_format='channels_last',
                      dilation_rate=(1, 1, 1),
                      activation='relu',
                      use_bias=True,
                      kernel_initializer='glorot_normal',
                      bias_initializer='zeros',
                      kernel_regularizer=l2(0.05),
                      bias_regularizer=l2(0.05),
                      activity_regularizer=None,
                      kernel_constraint=None,
                      bias_constraint=None)(x)
    window=[0,0,w1,w2,0]

dataset_add = '../../data/RFI_data_Sep_2019/prepared/'
n1,n2 = 40,20

def pp(X,Y):
    return process(X,Y,
                   mode=mode,
                   phase_in=phase_in,
                   phase_out=phase_out)

dp = Data_Provider_Sep_2019(nx=100,ny=200,
                            prefix=dataset_add,
                            window=window,
                            process=pp,
                            n_batch=10,
                            call_freq_train = 50,
                            call_freq_valid = 20,
                            split = [n1,n1+n2],
                            phase_in = phase_in,
                            phase_out = phase_out,
                            postprocess = None
                            )

dsh = dp.get_train(1)[1].shape
nx,ny,n_chan = dsh[1],dsh[2],dsh[-1]
pfix = 'try'+n_try+'_'
model_add = './models/'+pfix+str(mode)
ch_mkdir(model_add)

the_print('The model will be saved in:'+model_add,
          style='bold',
          tc='red',
          bgc='black')

x,y = dp.get_train(1)
print(x.shape,y.shape)
#input_img = Input(batch_shape=(None, 100, 4096, 1))
#x_in = tf.placeholder(tf.float32,[None]+list(x.shape[1:]))
x_in = Input(shape=list(x.shape[1:]))
y_true = tf.placeholder(tf.float32,[None]+list(y.shape[1:]))
learning_rate_var = tf.placeholder(tf.float32)

#\sum_ij (Y_ij - Y_predict,ij)^2 / N    
def loss_function(y_true,x_out):
#    loss = tf.reduce_mean(tf.pow(tf.log(y_true+1) - x_out, 2))
    loss = tf.reduce_mean(tf.pow(tf.log(y_true+1) - tf.log(x_out+1), 2))
#    loss = loss+tf.losses.get_regularization_loss()
    return loss

x0 = conv(x_in)
x1 = conv(x0)
x2 = conv(x1)
x3 = Add()([x2, x0])
#x4 = BatchNormalization()(x3,training=0)
x5 = conv(x3)
x_out = conv(x5,filters=n_chan)
print(y_true,x_out)
loss = loss_function(y_true,x_out)

model = Model(inputs=x_in, outputs=x_out)
model.summary()

optimizer = tf.train.RMSPropOptimizer(learning_rate_var).minimize(loss)
sess = tf.InteractiveSession()
saver = tf.train.Saver()

try:
    saver.restore(sess, model_add+'/model')
    the_print('Model is restored!',style='bold',tc='blue',bgc='black')
except:
    the_print('Something is wrong, model can not be restored!',style='bold',tc='red',bgc='black')
    init = tf.global_variables_initializer()
    sess.run(init)
#        exit()

#60,15,4,1
#8819MiB    1
#7795MiB    2
#5619MiB    3
#2547MiB    4
iterations = 10
if ns==0:
    auto_batch = [32,16,8,4]
    ns = auto_batch[mode-1]

epochs = []
losses = []
loss_val = []
losst_min = 1000

for epoch in range(training_epochs):
    # Loop over all batches
    learning_rate = learning_rate*decay_rate
    cc = 0
    ii = 0
    i = 0
    while True:
        while True:
            xb,yb = dp.get_train(ns)
            if xb is not None:
                break
        # Run optimization op (backprop) and loss op (to get loss value)
        o, c, pred = sess.run(fetches=[optimizer, loss, x_out],
                               feed_dict={x_in: xb, y_true: yb,
                                          learning_rate_var: learning_rate,
                                          K.learning_phase(): 1})
        i += 1
        cc += c
        ii += 1 
        if pred.std()==0:
            print('The model is dead!') 
            init = tf.global_variables_initializer()
            sess.run(init)
        if i > iterations: 
            break             
    
    # Display loss per epoch step
    if verbose:
        if epoch%verbose==0:
            print('Epoch:{:d}, lr= {:f}, loss= {:f}'.format(epoch,
                                                            learning_rate,
                                                            cc/ii))

            xb,yb = dp.get_valid()
            losst = sess.run(loss,
                             feed_dict={x_in: xb,
                                        y_true: yb,
                                        K.learning_phase(): 0})
                                        
            print('Validation, epoch:{:d}, loss= {:f}'.format(epoch, losst))

            if losst<losst_min:
                losst_min = losst
                saver.save(sess, model_add+'/best_model')      
                model.save(model_add+'/best_model.h5')  
                model.save_weights(model_add+'/best_model_weights.h5')
            

        epochs.append(epoch)
        losses.append(cc/ii)
        loss_val.append(losst)
        
# Creates a saver.
saver.save(sess, model_add+'/model')
model.save(model_add+'/latest_model.h5')
model.save_weights(model_add+'/latest_model_weights.h5')

try:
    ddd = np.load(model_add+'/log.npz')
    epochs0 = list(ddd['epochs'])
    losses0 = list(ddd['losses'])
    loss_val0 = list(ddd['loss_val'])
    epochs0.extend(list(epochs))
    losses0.extend(list(losses))
    loss_val0.extend(list(loss_val))
    epochs = epochs0
    losses = losses0
    loss_val = loss_val0
except:
    pass

np.savez(model_add+'/log',epochs=epochs,losses=losses,loss_val=loss_val)

plt.plot(losses)
plt.plot(loss_val)
plt.yscale('log')
plt.savefig(model_add+'/log.jpg')

#else:
#    dptest = Test_Provider_Jul_2019(n_ch-1)
#    x_in = tf.placeholder(tf.float32,[None,dptest.nx, dptest.ny, dptest.n_chan])
#    y_true = tf.placeholder(tf.float32,[None,dptest.nx, dptest.ny, 1])
#    print('test:')

#    loss_list = []
#    loopc = 1
#    while loopc==1:
#        data,rfi = dptest.next()
#        if data is None:
#            break
#        losst = sess.run([loss],
#                       feed_dict={x_in: data, y_true: rfi,learning_rate_var: learning_rate,
#                        K.learning_phase(): 0})
#        loss_list.append(losst)
#    print(np.mean(loss_list),np.std(loss_list))  




























