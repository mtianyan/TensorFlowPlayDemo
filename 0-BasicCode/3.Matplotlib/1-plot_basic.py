# 引入 Matplotlib 的分模块 Pyplot
import matplotlib.pyplot as plt
import numpy as np

'''
创建数据,-2到2区间的100个线性数据作为x.
绘制线性函数，三次函数图像。
'''
x = np.linspace(-2, 2, 100)
y1 = 3 * x + 4
y2 = x ** 3

# 创建图像
plt.plot(x, y1)
plt.plot(x, y2)

# 显示图像
plt.show()
