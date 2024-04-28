import numpy as np
from Calculate_differentiation import dW_differentiation


#计算dw的方法
def dw():
    # 为了'原始数据值@权重=目标值'，要进行梯度下降，要求dw1，以让'(w1-dw1)@x1=target_z'，如何求dw1是以下要研究的内容
    # 计算'(w1-dw1)@x1=target_z'，根据矩阵乘法分配律:
    # (w1-dw1)@x1=target_z   ===>  w1@x1-dw1@x1=target_z   ===>  w1@x1-dw1@x1=w1@x1-deta_z   ===>  dw1@x1=deta_z
    # 因此，如果求出的dw1符合'dw1@x1=deta_z'，那么此dw1就是完全精确的，否则是错误的或者只是近似值

    w1 = np.array([1, 3, 4, 2, 5, 2]).reshape(2, 3)#权重
    x1 = np.array([1, 2, 3,1, 2, 3]).reshape(3, 2)#原始数据值
    target_z=np.array([9,9,9,9]).reshape(2, 2)#目标值

    print('w1:','\n',w1,'\n')
    print('x1:','\n',x1,'\n')
    print('x1.T:','\n',x1.T,'\n')
    z1=w1@x1#计算值
    print('z1:','\n',z1,'\n')



    deta_z=z1-target_z#计算值与目标值的偏差
    print('deta_z:','\n',deta_z,'\n')






    # 计算dw1


    #方法一
    dw1=deta_z@x1.T
    print('dw1:','\n',dw1,'\n')
    dw1_z1=dw1@x1/25#数值统一设定缩小25倍
    print('dw1_z1:','\n',dw1_z1,'\n')


    #方法二
    dW_diff=dW_differentiation()
    w1_grad = dW_diff.zhuanzhi_hl_sum_numpy(w1, x1, 'zl')
    print('w1_grad:','\n',w1_grad,'\n')
    xin_w2=w1_grad*deta_z[:,0].reshape(2,1)
    xin_w3=w1_grad*deta_z[:,1].reshape(2,1)
    print('xin_w2   xin_w3:'  ,'\n',  xin_w2,'\n',xin_w3,'\n')
    q1=(xin_w2@x1[:,0]).reshape(2,1)
    print(q1)
    q2=(xin_w3@x1[:,1]).reshape(2,1)
    print(q2)
    dw1_z2 = np.concatenate((q1, q2), axis=1)/25#数值统一设定缩小25倍
    dw1_z2 = dW_diff.divide_by_2_until_odd_numpy(dw1_z2)
    print('dw1_z2:','\n',dw1_z2,'\n')


    dw1_z2是否与deta_z为倍数关系=dw1_z2/deta_z
    print('dw1_z2是否与deta_z为倍数关系:','\n',dw1_z2是否与deta_z为倍数关系,'\n')
    dw1_z1是否与deta_z为倍数关系=dw1_z1/deta_z
    print('dw1_z1是否与deta_z为倍数关系:','\n',dw1_z1是否与deta_z为倍数关系,'\n')

    # 结论：
    # 方法一计算的dw1只是近似值，在学习率很小的时候逐步梯度下降，精度适用，计算简洁方便
    # 方法二计算的dw1才是真值，不过计算复杂一点
    # 方法二输出：
    #  [[1. 1.]
    #  [1. 1.]]
    # 代表dw1_z2=deta_z，即方法二的dw1@x1=deta_z√


if __name__=='__main__':
    dw()