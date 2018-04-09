# -*- coding: UTF-8 -*-

"""
RNN-LSTM 循环神经网络
"""

import tensorflow as tf
import keras

# 神经网络的模型
def network_model(inputs, num_pitch, weights_file=None):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(
        512, # 输出的维度
        input_shape=(inputs.shape[1], inputs.shape[2]), # 输入的形状
        return_sequences=True # 返回 Sequences（序列）
    ))
    model.add(keras.layers.Dropout(0.3)) # 丢弃 30%
    model.add(keras.layers.LSTM(512, return_sequences=True))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.LSTM(512))
    model.add(keras.layers.Dense(256)) # 256 个神经元的全连接层
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(num_pitch)) # 所有不重复的音调的数目
    model.add(keras.layers.Activation('softmax')) # Softmax 激活函数算概率
    # 交叉熵计算误差，使用 RMSProp 优化器
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    if weights_file is not None: # 如果是 生成 音乐时
        # 从 HDF5 文件中加载所有神经网络层的参数（Weights）
        model.load_weights(weights_file)

    return model
