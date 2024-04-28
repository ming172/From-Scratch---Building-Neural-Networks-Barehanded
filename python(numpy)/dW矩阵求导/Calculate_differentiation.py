import numpy as np

# 计算矩阵微分
class dW_differentiation:
    def __init__(self):
        pass

    def zhuanzhi_hl_sum_numpy(self,tensor1, tensor2, hl):
        # 这个函数实现的功能：
        # 输入两个矩阵，两个矩阵相乘结果为z，其中一个矩阵要求梯度，另一个不要，
        # 将不要求梯度的矩阵转置后，让其每行/列求和，即可求得z对于要求梯度的矩阵的梯度，
        # 参数tensor1,tensor2为点乘的顺序，必须是tensor1@tensor2，
        # hl为哪个矩阵要求梯度，如果是左边的tensor1，则hl=='zl'左行，如果是右边的tensor2，则hl=='yh'右列
        # hl=='zl'时，函数的功能，相当于是计算tensor2转置后每一行相加的和按照tensor1的结构重新构成的矩阵，这个重新构成的矩阵，就是z对于tensor1矩阵的梯度

        if hl == 'yh':#左行
            sums = np.sum(tensor1, axis=0).reshape(1, -1).T
            new_matrix = np.tile(sums, (1, tensor2.shape[1]))

        if hl == 'zl':#右列
            sums = np.sum(tensor2, axis=1).reshape(-1, 1).T
            new_matrix = np.tile(sums, (tensor1.shape[0],1 ))

        return new_matrix

    def divide_by_2_until_odd_numpy(self,array):
        result = array.copy()  # 复制输入数组，避免修改原始数组
        while np.all(result % 2 == 0):  # 循环直到整个数组中的任一元素都无法再被2整除
            result = np.floor_divide(result, 2)  # 将整个数组中的所有元素都除以2
            if np.any(result  == 0):
                break
        return result

        # # 测试函数
        # a = np.array([[40., 60., 80.],
        #               [96., 144., 192.]])
        # a1 = np.array([[10., 15., 20.],
        #                [24., 36., 48.]])
        # result = divide_by_2_until_odd_numpy(a)
        # print(result)
        # print(np.array_equal(result, a1))  # 检查结果是否与预期一致

