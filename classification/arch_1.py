
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf


from tensorflow.keras.layers import BatchNormalization, Conv2D, Add, Activation
from tensorflow.keras.layers import Conv3D, Concatenate, Input, Lambda
from tensorflow.keras.backend import clip
from tensorflow.keras.regularizers import l2
from keras.losses import binary_crossentropy

def conv(x,kernel_size=(5,5),filters=12, activation='linear'):
    return Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=(1,1),
                  activation=activation,
                  padding='same',
#                  kernel_initializer='glorot_normal',
#                  bias_initializer='zeros',
                  kernel_regularizer=l2(0.001),
                  bias_regularizer=l2(0.001)
                  )(x)

def cba(x,kernel_size=(5,5),filters=12, activation='relu'):
    x = conv(x,kernel_size=(kernel_size),filters=filters)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def cab(x,kernel_size=(5,5),filters=12, activation='relu'):
    x = conv(x,kernel_size=(kernel_size),filters=filters)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)   
    return x

#def architecture(x_in):

#	xp = conv(x_in, activation='relu')
#	x1 = conv(xp, activation='relu')
#	x2 = conv(x1, activation='relu')
#	x3 = x2+xp
#	x3 = BatchNormalization()(x3)
#	x4 = conv(x3, activation='relu')

#	x_out = conv(x4,filters=1,activation='relu')

#	return x_out

def architecture(x_in, trainable=1):

	xp = tf.layers.conv2d(x_in,filters=12,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu, trainable=trainable)
	x1 = tf.layers.conv2d(xp,filters=12,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu, trainable=trainable)
	x2 = tf.layers.conv2d(x1,filters=12,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu, trainable=trainable)
	x3 = x2+xp
	x3 = tf.layers.batch_normalization(x3)
	x4 = tf.layers.conv2d(x3,filters=12,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu)

	x_out = tf.layers.conv2d(x4,filters=1,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu)

	return x_out
