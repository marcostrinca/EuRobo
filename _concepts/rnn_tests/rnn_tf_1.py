import tensorflow as tf

n_inputs = 3
n_nerons = 5

x0 = tf.placeholder(tf.float32, [None, n_inputs])
x1 = tf.placeholder(tf.float32, [None, n_inputs])

Wx = tf.Variable(tf.random_normal(shape=[n_inputs,n_nerons], dtype=tf.float32))
Wy = tf.Variable(tf.random_normal(shape=[n_inputs,n_nerons], dtype=tf.float32))

b = tf.Variable(tf.zeros([1, n_nerons],dtype=tf.float32))

Y0 = tf.tanh(tf.matmul(x0, Wx) + b)
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(x1, Wx) + b)

init = tf.global_variables_initializer()

import numpy as np

#Mini-batch: instance 0, 1, 2, 3
x0_batch = np.array([[0,1,2], [3, 4, 5], [6, 7, 8], [9,0,1]]) #t=0
x1_batch = np.array([[9,8,7], [0,0,0], [6,5,4], [3,2,1]]) #t=1

with tf.Session() as sess:
	init.run()
	Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={x0: x0_batch, x1: x1_batch})

print(Y0_val)
print(Y1_val)