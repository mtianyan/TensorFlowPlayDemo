# -*- coding: UTF-8 -*-

'''
用梯度下降的优化方法来快速解决线性回归问题
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 构建数据：100个随机点
points_num = 100
# 之后要往vectors中填充100个点的值
vectors = []
# 用 Numpy 的正态随机分布函数生成 100 个点
# 这些点的(x, y)坐标值: 对应线性方程 y = 0.1 * x + 0.2
# 权重 (Weight) 为 0.1，偏差 (Bias)为 0.2
try:
    # 运行100次
    for i in xrange(points_num):
        # 横坐标值，随机正态分布函数。区间0-0.66
        x1 = np.random.normal(0.0, 0.66)
        # 在真实值上加一些偏差
        y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0, 0.04)
        # 将点list加入vectors列表中
        vectors.append([x1, y1])
except:
    for i in range(points_num):
        x1 = np.random.normal(0.0, 0.66)
        y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0, 0.04)
        vectors.append([x1, y1])

x_data = [v[0] for v in vectors] # 列表生成式取出真实的点的 x 坐标
y_data = [v[1] for v in vectors] # 真实的点的 y 坐标

# 图像 1 ：展示 100 个随机数据点
plt.plot(x_data, y_data, 'ro', label="Original data") # 红色星形的点
plt.title("Linear Regression using Gradient Descent")
# 展示label
plt.legend()
plt.show()

# 构建线性回归模型

# 初始化参数，传入shape，最小值，最大值
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # 初始化 Weight
# 偏差。初始化为0
b = tf.Variable(tf.zeros([1]))                     # 初始化 Bias
# 这里的y是y帽子，也就是模型预测出来的值
y = W * x_data + b                                 # 模型计算出来的 y

# 定义 loss function (损失函数) 或 cost function (代价函数)
# 计算残差平方和。用(y帽子-真实的y)的平方累加的和。N就是总的点数，100.
# 对 Tensor 的所有维度计算 ((y - y_data) ^ 2) 之和 / N

# reduce_mean就是最后面的/N操作。square平方: y - y_data
loss = tf.reduce_mean(tf.square(y - y_data))

# 用梯度下降的优化器来优化我们的 loss functioin
# 让它更快的找到最终最拟合的w和b: 梯度下降的优化器。学习率，梯度下降的快慢。
optimizer = tf.train.GradientDescentOptimizer(0.5) # 设置学习率为 0.5(步长)，一般都是小于1的数。
# 太大的学习率可能错过局部最小值的那个点。
# 让它(损失函数)尽可能的损失最小
train = optimizer.minimize(loss)

# 创建会话
sess = tf.Session()

# 初始化数据流图中的所有变量
init = tf.global_variables_initializer()
sess.run(init)


try:
    # 训练 20 步
    for step in range(20):
        # 优化每一步
        sess.run(train)
        # 打印出每一步的损失，权重和偏差.必须run才能得到实际的值。
        print(("Step=%d, Loss=%f, [Weight=%f Bias=%f]") % (step, sess.run(loss), sess.run(W), sess.run(b)))
except:
    # 训练 20 步
    for step in xrange(20):
        # 优化每一步
        sess.run(train)
        # 打印出每一步的损失，权重和偏差
        print("Step=%d, Loss=%f, [Weight=%f Bias=%f]") \
                % (step, sess.run(loss), sess.run(W), sess.run(b))   



# 图像 2 ：绘制所有的点并且绘制出最佳拟合的直线
plt.plot(x_data, y_data, 'bo', label="Original data") # 蓝色圆圈的点
plt.title("Linear Regression using Gradient Descent")
# 横坐标是x_data.纵坐标为此时的wb确定的y
plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label="Fitted line") # 拟合的线
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 关闭会话
sess.close()
