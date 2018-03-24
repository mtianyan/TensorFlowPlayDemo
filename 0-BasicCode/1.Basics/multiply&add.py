# -*- coding: UTF-8 -*-

# 引入 TensorFlow 库
import tensorflow as tf

# 设置了gpu加速提示信息太多了，设置日志等级屏蔽一些
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

a = tf.constant(2)
b = tf.constant(3)
c = tf.multiply(a,b)
d = tf.add(c, 1)

with tf.Session() as sess:
    print (sess.run(d))