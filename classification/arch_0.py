
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf
    
def architecture(x_in):

	xp = tf.layers.conv2d(x_in,filters=12,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu)
	x1 = tf.layers.conv2d(xp,filters=12,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu)

	x_out = tf.layers.conv2d(x1,filters=1,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu)

	return x_out
