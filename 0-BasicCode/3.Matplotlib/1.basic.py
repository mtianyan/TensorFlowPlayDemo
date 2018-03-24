# -*- coding: UTF-8 -*-

# 引入 Matplotlib 的分模块 pyplot
import matplotlib.pyplot as plt
# 引入 numpy
import numpy as np

# 创建数据
x = np.linspace(-2, 2, 100)
#y = 3 * x + 4
y1 = 3 * x + 4
y2 = x ** 3

# 创建图像
#plt.plot(x, y)
plt.plot(x, y1)
plt.plot(x, y2)

# 显示图像
plt.show()
