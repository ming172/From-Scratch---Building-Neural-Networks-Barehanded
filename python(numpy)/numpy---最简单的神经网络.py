import numpy as np

def ReLU(z):
    return np.maximum(z, 0)

def ReLU_deriv(z):
    return z > 0

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = Z2
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1,A0,A2,Y):
    dZ2 = (A2 - Y)
    dA1 = A1.T @ dZ2
    dZ1 = dA1 * ReLU_deriv(Z1)
    dW1 = dZ1 @ A0.T
    db1 = dZ1
    dW2 = dZ2 @ A1.T
    db2 = dZ2
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def init_params():
    W1 = np.random.rand(2, 3) - 0.5
    b1 = np.random.rand(2, 2) - 0.5
    W2 = np.random.rand(2, 2) - 0.5
    b2 = np.random.rand(2, 2) - 0.5
    return W1, b1, W2, b2

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def gradient_descent(X, Y, alpha, iterations, threshold=1e-18):
    W1, b1, W2, b2=init_params()
    for i in range(1,iterations+1):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1,A0,A2,Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 100== 0:
            print("Iteration: ", i)
            result=W2@ReLU(W1@X+b1)+b2
            print("结果: ",result )
            loss = mse_loss(result, Y)
            print("损失: ", loss)
            if loss < threshold:
                print("达到所需精度，结束训练。",'\n')
                break
    return W1, b1, W2, b2


#学习率
alpha=0.001

#通过numpy构建的神经网络拟合函数：输入A0矩阵，输出q1矩阵
A0 =np.array([1,2,3,4,5,6]).reshape(3,2)
# q1=[4,578,35,25]
q1=[    414    ,  538    ,     43     ,    1    ]
Y=np.array([q1[0],q1[1],q1[2],q1[3]]).reshape(2,2)
W1, b1, W2, b2 = gradient_descent(A0, Y, alpha, 30000)
a=[W1, b1, W2, b2]
print('训练得到的W1, b1, W2, b2参数分别是：')
for i in a:
    print(i)
