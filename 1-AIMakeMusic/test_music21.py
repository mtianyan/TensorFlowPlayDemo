# -*- coding: UTF-* -*-

from music21 import converter, instrument

def print_notes():
    # 读取 MIDI 文件, 输出 Stream 流类型
    stream = converter.parse("qinghuaci.mid")

    # 获得所有乐器部分
    parts = instrument.partitionByInstrument(stream)

    if parts: # 如果有乐器部分，取第一个乐器部分
        print(parts.parts.srcStreamElements) #__dict__
        notes = parts.parts[0].recurse()
        print('*******')
    else:
        notes = midi.flat.notes

    # 打印出每一个元素
    for element in notes:
        print(str(element))

if __name__ == "__main__":
    print_notes()
