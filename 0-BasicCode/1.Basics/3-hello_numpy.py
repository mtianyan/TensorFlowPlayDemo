import numpy as np

# 创建一维数组(数组)
vector = np.array([1, 2, 3])
print("shape: ", vector.shape)
print("size: ", vector.size)
print("dim: ", vector.ndim)
print(type(vector))

# 创建二维数组(矩阵)
matrix = np.array([[1, 2], [3, 4]])
print("shape: ", matrix.shape)
print("size: ", matrix.size)
print("dim: ", matrix.ndim)
print(type(matrix))
