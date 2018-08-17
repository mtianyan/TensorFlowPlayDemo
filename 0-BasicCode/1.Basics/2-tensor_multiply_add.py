# -*- coding: UTF-8 -*-
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
# TensorFlow中的乘法
c = tf.multiply(a, b)
# TensorFlow中的加法
d = tf.add(c, 1)

with tf.Session() as sess:
    print("2*3+1 = ", end='')
    print(sess.run(d))
