# 这个代码没有实际的意义，就是为了方便检验和查看各种npy文件
import numpy as np

# 假设你的文件名是 'data.npy'
data = np.load('pred_label/CNN_labels.npy')

# 打印数据以查看其内容
print(data)

# 打印数组的形状和数据类型
print("Shape of the array:", data.shape)
print("Data type of the array:", data.dtype)