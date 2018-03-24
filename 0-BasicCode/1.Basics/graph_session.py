# -*- coding: UTF-8 -*-

# 引入tensorflow
import tensorflow as tf

# 设置了gpu加速提示信息太多了，设置日志等级屏蔽一些
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 创建两个常量 Tensor.第一个为1行2列，第二个为二行一列。
# 也就是矩阵乘法必须满足，列等于行。
const1 = tf.constant([[2, 2]])
const2 = tf.constant([[4],
                      [4]])

# 矩阵乘法运算matrix mul tf.add()
multiple = tf.matmul(const1, const2)

# 尝试用print输出multiple的值, 不会输出真实值。因为没运行
print(multiple)

# 创建了 Session (会话) 对象
sess = tf.Session()

# 用Session的run方法来实际运行multiple这个矩阵乘法操作
# 并把操作执行的结果赋值给 result
result = sess.run(multiple)

# 用print打印矩阵乘法的结果
print(result)

if const1.graph is tf.get_default_graph():
    print("const1所在的图（Graph）是当前上下文默认的图")

# 关闭已用完的Session(会话)
sess.close()

# 第二种方法来创建和关闭Session, with上下文管理器
with tf.Session() as sess:
    result2 = sess.run(multiple)
    # 第一个参数为指定的保存路径，第二个参数为要保存的图
    tf.summary.FileWriter("./", sess.graph)
    print("Multiple的结果是 %s " % result2)
