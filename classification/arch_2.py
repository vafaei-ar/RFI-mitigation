
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

def architecture(x_in, trainable=1):

    xp = tf.layers.conv2d(x_in,filters=12,
                          kernel_size=5,
                          strides=(1, 1),
                          padding='same',
                          trainable=trainable)
    x = tf.layers.batch_normalization(xp)
    x = tf.nn.relu(x)
    x1 = tf.layers.conv2d(x,filters=12,
                          kernel_size=5,
                          strides=(1, 1),
                          padding='same',
                          trainable=trainable)
    x1 = tf.layers.batch_normalization(x1)
    x1 = tf.nn.relu(x1)
    x2 = tf.layers.conv2d(x1,filters=12,
                          kernel_size=5,
                          strides=(1, 1),
                          padding='same',
                          trainable=trainable)
    x2 = tf.layers.batch_normalization(x2)
    x2 = tf.nn.relu(x2)

    x3 = x2+xp

    x4 = tf.layers.conv2d(x3,filters=12,kernel_size=5,strides=(1, 1),padding='same')
    x4 = tf.layers.batch_normalization(x4)
    x4 = tf.nn.relu(x4)
    
    x5 = tf.layers.conv2d(x4,filters=12,kernel_size=5,strides=(1, 1),padding='same')
    x5 = tf.layers.batch_normalization(x5)
    x5 = tf.nn.relu(x5)

    x_out = tf.layers.conv2d(x5,filters=1,kernel_size=5,strides=(1, 1),padding='same',
            activation=tf.nn.relu)

    return x_out

#def architecture(x_in):

#    xp = tf.layers.conv2d(x_in,filters=12,
#        kernel_size=5,
#        padding='SAME',
#        kernel_initializer=tf.contrib.layers.xavier_initializer(),
#        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.05),
#        bias_initializer=tf.constant_initializer(0.1),
#        name='conv_1')
#    x = tf.layers.batch_normalization(xp,name='norm_1')
#    x = tf.nn.relu(x,name='act_1')
#    
#    x1 = tf.layers.conv2d(x,filters=12,
#        kernel_size=5,
#        padding='SAME',
#        kernel_initializer=tf.contrib.layers.xavier_initializer(),
#        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.05),
#        bias_initializer=tf.constant_initializer(0.1),
#        name='conv_2')
#    x1 = tf.layers.batch_normalization(x1,name='norm_2')
#    x1 = tf.nn.relu(x1,name='act_2')
#    
#    x2 = tf.layers.conv2d(x1,filters=12,
#        kernel_size=5,
#        padding='SAME',
#        kernel_initializer=tf.contrib.layers.xavier_initializer(),
#        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.05),
#        bias_initializer=tf.constant_initializer(0.1),
#        name='conv_3')
#    x2 = tf.layers.batch_normalization(x2,name='norm_3')
#    x2 = tf.nn.relu(x2,name='act_3')

#    x3 = x2+xp

#    x4 = tf.layers.conv2d(x3,filters=12,
#        kernel_size=5,
#        padding='SAME',
#        kernel_initializer=tf.contrib.layers.xavier_initializer(),
#        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.05),
#        bias_initializer=tf.constant_initializer(0.1),
#        name='conv_4') 
#    x4 = tf.layers.batch_normalization(x4,name='norm_4')
#    x4 = tf.nn.relu(x4,name='act_4')
#    
#    x5 = tf.layers.conv2d(x4,filters=12,
#        kernel_size=5,
#        padding='SAME',
#        kernel_initializer=tf.contrib.layers.xavier_initializer(),
#        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.05),
#        bias_initializer=tf.constant_initializer(0.1),
#        name='conv_5')
#    x5 = tf.layers.batch_normalization(x5,name='norm_5')
#    x5 = tf.nn.relu(x5,name='act_5')

#    x5 = tf.layers.conv2d(x5,filters=1,kernel_size=5,strides=(1, 1),padding='same',name='conv_6')
#    x_out = tf.nn.relu(x5,name='act_6')

#    return x_out
