# -*- coding: UTF-8 -*-

"""
训练神经网络，将参数（Weight）存入 HDF5 文件
"""

import numpy as np
import tensorflow as tf
import keras

from utils import *
from network import *

# 训练神经网络
def train():
    notes = get_notes()

    # 得到所有不重复（因为用了set）的音调数目
    num_pitch = len(set(notes))

    network_input, network_output = prepare_sequences(notes, num_pitch)

    model = network_model(network_input, num_pitch)

    filepath = "weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=66666, batch_size=64, callbacks=callbacks_list)

def prepare_sequences(notes, num_pitch):
    """
    为神经网络准备好供训练的序列
    """
    sequence_length = 100  # 序列长度

    # 得到所有音调的名字
    pitch_names = sorted(set(item for item in notes))

    # 创建一个字典，用于映射 音调 和 整数
    pitch_to_int = dict((pitch, num) for num, pitch in enumerate(pitch_names))

    network_input = []
    network_output = []

    # 创建输入序列，以及对应的输出序列
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([pitch_to_int[char] for char in sequence_in])
        network_output.append(pitch_to_int[sequence_out])

    n_patterns = len(network_input)

    # 将输入的形状转换成 LSTM 模型可以接受的
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # 将输入标准化
    network_input = network_input / float(num_pitch)

    # 转换成 {0, 1} 组成的布尔矩阵，为了配合 categorical_crossentropy 误差算法使用
    network_output = keras.utils.to_categorical(network_output)

    return (network_input, network_output)

if __name__ == '__main__':
    train()
