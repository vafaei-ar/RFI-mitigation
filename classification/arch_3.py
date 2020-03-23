
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

def architecture(x_in):

    xp = tf.layers.conv2d(x_in,filters=12,kernel_size=5,strides=(1, 1),padding='same')
    x = tf.layers.batch_normalization(xp)
    x = tf.nn.relu(x)
    x1 = tf.layers.conv2d(x,filters=12,kernel_size=5,strides=(1, 1),padding='same')
    x1 = tf.layers.batch_normalization(x1)
    x1 = tf.nn.relu(x1)
    x2 = tf.layers.conv2d(x1,filters=12,kernel_size=5,strides=(1, 1),padding='same')
    x2 = tf.layers.batch_normalization(x2)
    x2 = tf.nn.relu(x2)

    x3 = x2+xp

    x4 = tf.layers.conv2d(x3,filters=12,kernel_size=5,strides=(1, 1),padding='same')
    x4 = tf.layers.batch_normalization(x4)
    x4 = tf.nn.relu(x4)
    
    x6 = tf.layers.conv2d(x4,filters=12,kernel_size=5,strides=(1, 1),padding='same')
    x6 = tf.layers.batch_normalization(x6)
    x6 = tf.nn.relu(x6)
    
    x7 = x6+x3

    x8 = tf.layers.conv2d(x7,filters=12,kernel_size=5,strides=(1, 1),padding='same')
    x8 = tf.layers.batch_normalization(x8)
    x8 = tf.nn.relu(x8)

    x_out = tf.layers.conv2d(x8,filters=1,kernel_size=5,strides=(1, 1),padding='same',
            activation=tf.nn.relu)

    return x_out
