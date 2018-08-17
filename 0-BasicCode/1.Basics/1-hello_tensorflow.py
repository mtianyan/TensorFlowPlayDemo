# 引入 TensorFlow 库
import tensorflow as tf

'''
设置了gpu加速提示信息太多了，设置日志等级屏蔽一些,里面有一些提示是因为你安装的是预编译好的TensorFlow。
下面两行命令可以让你像鸵鸟一样埋进沙子里，如果精力充沛，推荐源码编译安装
'''
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 创建一个常量 Operation (操作)
hw = tf.constant("Hello World! Mtianyan love TensorFlow!")

# 启动一个 TensorFlow 的 Session (会话)
sess = tf.Session()

# 运行 Graph (计算图)
print(sess.run(hw))

# 关闭 Session（会话）
sess.close()
