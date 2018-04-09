# -*- coding: UTF-8 -*-

"""
用训练好的神经网络模型来作曲
"""

import pickle
import numpy as np
from music21 import instrument, note, stream, chord
import tensorflow as tf

from utils import *
from network import *

# 以之前训练所得的最佳参数来生成音乐
def generate():
    # 加载用于训练神经网络的音乐数据
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # 得到所有音调的名字
    pitch_names = sorted(set(item for item in notes))

    # 得到所有不重复（因为用了set）的音调数目
    num_pitch = len(set(notes))

    network_input, normalized_input = prepare_sequences(notes, pitch_names, num_pitch)

    # 载入之前训练时最好的参数，来生成神经网络模型
    model = network_model(normalized_input, num_pitch, "best-weights.hdf5")

    # 用神经网络来生成音乐数据
    prediction = generate_notes(model, network_input, pitch_names, num_pitch)

    # 用音乐数据生成 MIDI 文件，再转换成 MP3
    create_music(prediction)

def prepare_sequences(notes, pitch_names, num_pitch):
    """
    为神经网络准备好供训练的序列
    """
    # 创建一个字典，用于映射 音调 和 整数
    pitch_to_int = dict((pitch, number) for number, pitch in enumerate(pitch_names))

    sequence_length = 100
    
    network_input = []
    output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([pitch_to_int[char] for char in sequence_in])
        output.append(pitch_to_int[sequence_out])

    n_patterns = len(network_input)

    # 将输入的形状转换成 LSTM 模型可以接受的
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # 将输入标准化
    normalized_input = normalized_input / float(num_pitch)

    return (network_input, normalized_input)

def generate_notes(model, network_input, pitch_names, num_pitch):
    """
    基于一序列音符，用神经网络来生成新的音符
    """

    # 从输入里随机选择一个序列，作为"预测"/生成的音乐的起始点
    start = np.random.randint(0, len(network_input) - 1)

    # 创建一个字典，用于映射 整数 和 音调
    int_to_pitch = dict((number, pitch) for number, pitch in enumerate(pitch_names))

    pattern = network_input[start]
    prediction_output = []

    # 生成 500 个 音符/音调
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(num_pitch)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_pitch[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

if __name__ == '__main__':
    generate()
