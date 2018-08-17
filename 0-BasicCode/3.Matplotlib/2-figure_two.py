import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(-4, 4, 50)
y1 = 3 * x + 2
y2 = x ** 2

# 第一张图，指定图的大小
plt.figure(num=1, figsize=(7, 6))

# 第一张图两个线
plt.plot(x, y1)
plt.plot(x, y2, color="red", linewidth=3.0, linestyle="--")

# 第二张图
plt.figure(num=2)
plt.plot(x, y2, color="green")

# 显示所有图像
plt.show()
