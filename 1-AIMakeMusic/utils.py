# -*- coding: UTF-8 -*-

import os
import subprocess
import pickle
import glob
from music21 import converter, instrument, note, chord,stream

def convertMidi2Mp3():
    """
    将神经网络生成的 MIDI 文件转成 MP3 文件
    """
    input_file = 'output.mid'
    output_file = 'output.mp3'

    assert os.path.exists(input_file)

    print('Converting %s to MP3' % input_file)

    # 用 timidity 生成 mp3 文件
    command = 'timidity {} -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 128k {}'.format(input_file, output_file)
    subprocess.call(command, shell=True)

    print('Converted. Generated file is %s' % output_file)

def get_notes():
    """
    从 music_midi 目录中的所有 MIDI 文件里提取 note（音符）和 chord（和弦）
    Note 样例： A, B, A#, B#, G#, E, ...
    Chord 样例: [B4 E5 G#5], [C5 E5], ...
    因为 Chord 就是几个 Note 的集合，所以我们把它们简单地统称为“Note”
    """
    notes = []

    # glob : 匹配所有符合条件的文件，并以 List 的形式返回
    for file in glob.glob("music_midi/*.mid"):
        stream = converter.parse(file)

        # 获取所有乐器部分
        parts = instrument.partitionByInstrument(stream)

        if parts: # 如果有乐器部分， 取第一个乐器部分
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = stream.flat.notes

        for element in notes_to_parse:
            # 如果是Note类型，那么取它的音调
            if isinstance(element, note.Note):
                # 格式例如： E6
                notes.append(str(element.pitch))
            # 如果是Chord类型，那么取它各个音调的序号
            elif isinstance(element, chord.Chord):
                # 转换后格式例如： 4.15.7
                notes.append('.'.join(str(n) for n in element.normalOrder))

    # 将数据写入 data/notes 文件
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def create_music(prediction):
    """
    用神经网络'预测'的音乐数据来生成 MIDI 文件，再转成 MP3 文件
    """
    offset = 0   # 偏移
    output_notes = []

    # 生成 Note（音符）或 Chord（和弦）对象
    for data in prediction:
        # 是 Chord。格式例如： 4.15.7
        if ('.' in data) or data.isdigit():
            notes_in_chord = data.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano() # 乐器用钢琴
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # 是 Note
        else:
            new_note = note.Note(data)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # 每次迭代都将偏移增加，这样才不会交叠覆盖
        offset += 0.5

    # 创建音乐流（Stram）
    midi_stream = stream.Stream(output_notes)

    # 写入 MIDI 文件
    midi_stream.write('midi', fp='output.mid')

    # 将生成的 MIDI 文件转换成 MP3
    convertMidi2Mp3()

if __name__ == '__main__':
    convertMidi2Mp3()