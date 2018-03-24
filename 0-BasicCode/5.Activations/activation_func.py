# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 创建输入数据
x = np.linspace(-7, 7, 180) # (-7, 7) 之间等间隔的 180 个点

# 激活函数的原始手工实现
def sigmoid(inputs):
    # y = 1 / (1 + exp(-x)) np.exp相当于e的多少次方
    y = [1 / float(1 + np.exp(-x)) for x in inputs]
    return y

def relu(inputs):
    # f(x) = max(0,x) x大于0时，函数值y就是x。x<0时函数值y就是0.
    # x如果大于0，则真值为1;y=x;而x若不满足>0真值为0;y=0
    y = [x * (x > 0) for x in inputs]
    return y

def tanh(inputs):
    # e的x次方-e的负x次方做分母。e的x次方+e的负x次方做分母
    y = [(np.exp(x) - np.exp(-x)) / float(np.exp(x) - np.exp(-x)) for x in inputs]
    return y

def softplus(inputs):
    # y = log(1+e的x平方)
    y = [np.log(1 + np.exp(x)) for x in inputs]
    return y

# 经过 TensorFlow 的激活函数处理的各个 Y 值
y_sigmoid = tf.nn.sigmoid(x)
y_relu = tf.nn.relu(x)
y_tanh = tf.nn.tanh(x)
y_softplus = tf.nn.softplus(x)

# 创建会话
sess = tf.Session()

# 运行run,得到四个返回值
y_sigmoid, y_relu, y_tanh, y_softplus = sess.run([y_sigmoid, y_relu, y_tanh, y_softplus])

# 创建各个激活函数的图像
plt.figure(1, figsize=(8, 6))

plt.subplot(221)
plt.plot(x, y_sigmoid, c='red', label='Sigmoid')
# y轴取值的区间
plt.ylim((-0.2, 1.2))
# 显示label，放在最适合的位置
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x, y_relu, c='red', label='Relu')
plt.ylim((-1, 6))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x, y_tanh, c='red', label='Tanh')
plt.ylim((-1.3, 1.3))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x, y_softplus, c='red', label='Softplus')
plt.ylim((-1, 6))
plt.legend(loc='best')

# 显示图像
plt.show()

# 关闭会话
sess.close()