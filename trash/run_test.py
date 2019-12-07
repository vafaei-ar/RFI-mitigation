import os
import argparse
import numpy as np

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf
print(tf.__version__)

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

parser.add_argument('--pin', required=False, help='theme', type=int, default=0)
parser.add_argument('--pout', required=False, help='theme', type=int, default=0)
parser.add_argument('--sep', required=False, help='theme', type=int, default=0)
parser.add_argument('--app', required=False, help='theme', type=int, default=1)

args = parser.parse_args()
mode = args.mode
w1 = args.w1
w2 = args.w2
n_try = args.ntry
phase_in = args.pin
phase_out = args.pout
sep = args.sep
app = args.app

if sep==0:
    p2 = None
elif sep and (mode==2 or mode==3):
    def p2(X,Y):
        return X[...,sep-1:sep],Y[...,sep-1:sep]
else:
    print('Seperation is only available for polarization!')
    exit()

dataset_add = '../../data/RFI_data_Sep_2019/prepared/'
n1,n2 = 40,20

def pp(X,Y):
    return process(X,Y,
                   mode=mode,
                   phase_in=phase_in,
                   phase_out=phase_out)

dp = Data_Provider_Sep_2019(nx=100,ny=200,
                            prefix=dataset_add,
                            process=pp,
                            n_batch=10,
                            call_freq_train = 50,
                            call_freq_valid = 1000,
                            split = [n1,n1+n2],
                            phase_in = phase_in,
                            phase_out = phase_out,
                            postprocess = p2
                            )

dsh = dp.get_train(1)[1].shape
nx,ny,n_chan = dsh[1],dsh[2],dsh[-1]
name = str(nx)+'x'+str(ny)
name = 'x'.join([str(i) for i in dsh[1:]])
name = ''
pfix = 'try'+n_try+'_'
if phase_in:
    pfix += 'in_'
if phase_out:
    pfix += 'out_'
if sep:
    pfix += str(sep)+'_'
if app!=1:
    pfix += 'app'+str(app)+'_'
model_add = './models/'+pfix+str(mode)#+'_'+name

x,y = dp.get_train(1)
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

if mode!=4:
    def conv(x,filters=12):
        return Conv2D(filters=filters,
                      kernel_size=(5,5),
                      strides=(1,1),
                      activation='relu',
                      padding='same',
#                      kernel_initializer='glorot_uniform',
#                      bias_initializer='zeros',
                      kernel_regularizer=l2(0.05),
                      bias_regularizer=l2(0.05))(x)
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
                      kernel_initializer='glorot_uniform',
                      bias_initializer='zeros',
                      kernel_regularizer=l2(0.05),
                      bias_regularizer=l2(0.05),
                      activity_regularizer=None,
                      kernel_constraint=None,
                      bias_constraint=None)(x)

x0 = conv(x_in)
x1 = conv(x0)
x2 = conv(x1)
x3 = Add()([x2, x0])
#    x4 = BatchNormalization()(x3,training=0)
x5 = conv(x3)
x_out = conv(x5,filters=n_chan)

loss = loss_function(y_true,x_out)
    
optimizer = tf.train.RMSPropOptimizer(learning_rate_var).minimize(loss)
sess = tf.InteractiveSession()
saver = tf.train.Saver()

#init = tf.global_variables_initializer()
#sess.run(init)

#saver.save(sess, 'xxx/model')
#saver.restore(sess, 'xxx/model')
#exit()

#saver.restore(sess, model_add+'/model')
#exit()

try:
    saver.restore(sess, model_add+'/best_model')
    the_print('Model is restored!',style='bold',tc='blue',bgc='black')
except:
    the_print('Something is wrong, model can not be restored!',style='bold',tc='red',bgc='black')
    exit()

loss_list = []
#while True:
#    data,rfi = dp.get_test()
#    try:
#        data,rfi = dp.get_test()
#    except IndexError:
#        break
#    except Exception as e:
#        print(e)
#        exit()

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

ch_mkdir(model_add+'/preds')
iii = 0

for data,rfi in dp:

    num = data.shape[0]
    inds = np.arange(num)
    for i in chunks(inds, 10):
        print(i)
#        losst = sess.run(loss,
#                         feed_dict={x_in: data[i],
#                                    y_true: rfi[i],
#                                    K.learning_phase(): 0})
#        loss_list.append(losst)
#        
#        pred = sess.run(x_out,
#                       feed_dict={x_in: data[i],
#                       K.learning_phase(): 0})

        losst,pred = sess.run(fetches=[loss,x_out],
                              feed_dict={x_in: data[i],
                                         y_true: rfi[i],
                                         K.learning_phase(): 0})
        loss_list.append(losst)
        print(rfi[i].shape,pred.shape)
        iii += 1
        np.savez(model_add+'/preds/'+str(iii),truth=rfi[i],pred=pred)
    
np.save(model_add+'/test',loss_list)




#    truth, pred = truth[0,:,:,0], pred[0,:,:,0]
#    losst[0,i] = np.mean(( np.log(truth+1)-np.log(pred+1) )**2)
#    truthp = truth+0
#    truthp[truthp==0] = 1e-50
#    truthp = np.log(truthp)
#    for j in range(100):
#        filt = (rngs[j]<=truthp) & (truthp<rngs[j+1])
#        truth2 = truth[filt]
#        pred2 = pred[filt]
#        loss[0,j,i] = np.sum(( np.log(truth2+1)-np.log(pred2+1) )**2)
#        num[0,j,i] = np.sum(filt)
#np.savez('temporal/'+key+'_res',loss = loss, num=num, losst = losst)
#        
        
        
        
        
        
        
        
        
        

























