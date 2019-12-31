"""
Detects RFI using a CNN (R-Net).
Has the ability to use Polerization and Baselines as extra channels to train on.
Trains on simulated data by Chris.

author: Alireza Vafaei Sadr
version: 18/12/19
"""
import os
import argparse


import matplotlib as mpl
import pylab as plt
import numpy as np
mpl.use('agg')


try:
    import tensorflow.compat.v1 as tf
    #Disable tensorflow v2 behavoirs
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

#Model groups layers into an object with training and inference features.
from tensorflow.keras.models import Model,load_model

#BatchNormailzation base class see: https://arxiv.org/pdf/1502.03167v3.pdf
#Conv2D to do 2D convolution: https://arxiv.org/pdf/1603.07285v2.pdf
#Add: Layer that adds a list of inputs.
from tensorflow.keras.layers import BatchNormalization, Conv2D, Add

#Concatenate: Layer that concatenates a list of inputs.
#Input(): Function to instantiate a Keras tensor.
from tensorflow.keras.layers import Conv3D, Concatenate, Input

#Regularizers allow to apply penalties on layer parameters or layer activity during optimization.
#See: https://keras.io/regularizers/
from tensorflow.keras.regularizers import l2

#Abstract Keras backend API
#See: https://keras.io/backend/
from tensorflow.keras import backend as K

###from keras.layers import Input, Dense, MaxPooling2D, MaxPooling3D, Dropout, BatchNormalization

#Import from utils.py script
from utils import *


def loss_function(y_true,x_out):
    """
    The loss function on which to optimize.

    Keyword arguments:
    y_true -- The ground truth
    x_out  -- Prediction
    """
    ###loss = tf.reduce_mean(tf.pow(tf.log(y_true+1) - x_out, 2))
    ###loss = loss+tf.losses.get_regularization_loss()

    loss = tf.reduce_mean(tf.pow(tf.log(y_true+1) - tf.log(x_out+1), 2))
    return loss



def conv_wrapper(mode, w1, w2):
    """Wraps convolution function depending on mode used."""

    if mode != 4:
        window=[0,w1,w2,0]
        def conv(x,filters=12):
                """
                First contsturct then use a 2D convolutional kernel on Tensor
                if not using Polerization and Baselines together

                Keyword arugments:
                x -- Tensor object on which to execute convolution on
                filters -- Number of filters (output channels) to be used
                """
                return Conv2D(filters=filters,
                              kernel_size=(5,5),
                              strides=(1,1),
                              activation='relu',
                              padding='same',
                              kernel_initializer='glorot_normal',
                              #bias_initializer='zeros',
                              kernel_regularizer=l2(0.05),
                              bias_regularizer=l2(0.05)
                             )(x)

    else:
        window=[0,0,w1,w2,0]
        def conv(x,filters=6):
            """
            Use a 3D convultional kernel if using Polerization and Baseline information together
            """
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
                          bias_constraint=None
                         )(x)
    return conv, window


def getArgs(passedArgs=None):
    """Gets and return command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=False,
                        help='Determines the method to predict RFI. mode=1 means inputs and outputs are frequency-time images.',
                        type=int, default=1)
    parser.add_argument('--w1', required=False, help='Determines the freq. size for the window of the frequency-time images.',
                        type=int, default=2000)
    parser.add_argument('--w2', required=False, help='Determines the freq. size for the window of the frequency-time images.',
                        type=int, default=50)
    parser.add_argument('--batch_size', required=False, help='Size of batch for the neural network.', type=int, default=0)
    parser.add_argument('--epochs', required=False,
                        help='Set the number of training \'epochs\'. Not strictly an epoch due to data augmentation',
                        type=int, default=25)
    parser.add_argument('--verbose', required=False, help='Display loss(es) per \'epoch\' step',
                        type=int, default=5)
    #Learning rate: https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10
    #Adaptive learning rates: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    parser.add_argument('--learning_rate', required=False, help='Set the learning rate',
                        type=float, default=0.05)
    parser.add_argument('--decay_rate', required=False, help='Set the decay rate',
                        type=float, default=0.985)
    parser.add_argument('--data_set_location', required=False, help='Set where to look for training data.',
                        type=str, default='../../data/prepared/')
    parser.add_argument('--ntry', required=True, help='Train different versions of a model, separates different versions.',
                        type=str)
    ###parser.add_argument('--ws', required=False, help='time window', type=int, default=100)
    return parser.parse_args(passedArgs)


def main(passedArgs=None):
    """
    TODO
    """
    args = getArgs(passedArgs)
    mode = args.mode
    w1 = args.w1
    w2 = args.w2
    n_try = args.ntry
    ns = args.batch_size
    training_epochs = args.epochs #Not actual epoch
    verbose = args.verbose
    learning_rate = args.learning_rate
    decay_rate = args.decay_rate
    dataset_add = args.data_set_location


    phase_in = False #? - What does phase in and phase out represent?
    phase_out = False
    #n1 = training set, n1 to n2 = validation set, n2 to end = test set
    n1,n2 = 40,20

    conv, window = conv_wrapper(mode, w1, w2)

    def pp(X,Y):
        """Wrap process function to use set mode and phases."""
        return process(X,Y, mode=mode, phase_in=phase_in, phase_out=phase_out)

    #Create a new data provider instance
    dp = Data_Provider_Sep_2019(prefix=dataset_add,
                                process=pp,
                                n_files=30,
                                window=window,
                                n_batch=10,
                                call_freq_train = 50,   #? - What is the purpose of these two parameters
                                call_freq_valid = 20,
                                #split = [n1,n1+n2],
                                phase_in = phase_in,
                                phase_out = phase_out,
                                postprocess = None)


    dsh = dp.get_train(1)[1].shape #dsh = data_shape?
    nx,ny,n_chan = dsh[1],dsh[2],dsh[-1] #? - nx, ny = 'windowed' dimensions respectively?

    pfix = 'try'+n_try+'_' #The trailing number is the mode used
    model_add = './models/'+pfix+str(mode)
    ch_mkdir(model_add)
    the_print('The model will be saved in:'+model_add,
              style='bold',
              tc='red',
              bgc='black')


    x,y = dp.get_train(1)
    print(x.shape,y.shape) #? - Does this still print out the self.n_time, self.n_baseline, self.n_freq, self.n_pol tuple?


    #input_img = Input(batch_shape=(None, 100, 4096, 1))
    #x_in = tf.placeholder(tf.float32,[None]+list(x.shape[1:]))
    x_in = Input(shape=list(x.shape[1:]))
    y_true = tf.placeholder(tf.float32,[None]+list(y.shape[1:]))
    learning_rate_var = tf.placeholder(tf.float32)


    #? - What's the point of using an R-Net type topology if we only have 6 layers?
    x0 = conv(x_in)
    x1 = conv(x0)
    x2 = conv(x1)
    x3 = Add()([x2, x0])
    #? - Why are we not using BatchNorm layer?
    ###x4 = BatchNormalization()(x3,training=0)
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
        ###exit()

    #These are size of batches on memory, I calculated them to find the best batch size according to GPU memory size.
    #60,15,4,1
    #8819MiB    1
    #7795MiB    2
    #5619MiB    3
    #2547MiB    4


    iterations = 10
    if ns==0:   #ns = number_of_samples? Is set as the batch size or auto
        auto_batch = [32,16,8,4]
        ns = auto_batch[mode-1]

    epochs = []
    losses = []
    loss_val = []
    losst_min = 1000

    #Start actuall training
    for epoch in range(training_epochs):    #? - Difference between an epoch and iteration? Why do it like this, for average loss?

        #Loop over all batches
        learning_rate = learning_rate*decay_rate #each 'epoch' it decays
        cc = 0  #Summation counter for the 'c' variable what does it represent? -> Looks like loss amount
        ii = 0  #? - Unclear why this ii variable is needed versus just using i? Since they both update at the same time
        i = 0   #? - See above

        #Seems like uses 'iterations' amount of calls per learning_rate which corresponds to an 'epoch'

        while i < iterations:
            xb, yb = dp.get_train(ns)
            while xb is None:
                xb,yb = dp.get_train(ns)  #Get train just return the training set and respective rfi (test) set

            # Run optimization op (backprop) and loss op (to get loss value)
            #? - What does 'o' represent and why is it not used?
            #? - Would 'current_loss' be a representative name for c?
            o, c, pred = sess.run(fetches=[optimizer, loss, x_out],
                                   feed_dict={x_in: xb, y_true: yb,
                                              learning_rate_var: learning_rate,
                                              K.learning_phase(): 1})
            i += 1
            ii += 1
            cc += c     #? - Likewise would 'cumalalitve_loss_per_epoch' be a representative name for cc?

            if pred.std()==0: #? - What does this mean?
                #? - If the model is re-initing for this epoch shouldn't cc, and ii (i) be set back to 0?
                print('The model is dead!')
                init = tf.global_variables_initializer()
                sess.run(init)


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
        # Load the saved npz file (with the different arrays)
        ddd = np.load(model_add+'/log.npz')
        # Load the arrays into differnt vars
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

    #Save several arrays into a single file in uncompressed .npz format:
    #Here model_add is is the folder where the model is stored under e.g. '/models/try1_1'
    #The other arguments are arrays specified to be saved. Each is stored as a key value pair in the .npz file
    np.savez(model_add+'/log',epochs=epochs,losses=losses,loss_val=loss_val)

    plt.plot(losses)
    plt.plot(loss_val)
    plt.yscale('log')
    plt.savefig(model_add+'/log.png')

if __name__ == '__main__':
    main()












##############################################--DEPRECIATED--##############################################
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




























