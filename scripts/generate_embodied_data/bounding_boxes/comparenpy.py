import numpy as np

# 加载两个 .npy 文件
array1 = np.load('1.npy')
array2 = np.load('2.npy')

# 比较两个数组是否相同
if np.array_equal(array1, array2):
    print("两个文件中的数值相同")
else:
    print("两个文件中的数值不同")