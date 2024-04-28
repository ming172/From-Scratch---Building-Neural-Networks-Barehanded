import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_data(filepath):
    data = pd.read_csv(filepath)
    data = np.array(data)
    np.random.shuffle(data)
    return data

def split_data(data,split_count):
    test = data[0:split_count].T
    train = data[split_count:].T
    return train,test

def preprocess_data(data):
    target = data[0]
    features = data[1:] / 255.
    return target, features

def init_params():
    W1 = np.random.rand(10, 784) - 0.5#-0.5是为了让数值在-0.5~0.5区间，后续可以ReLU()
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def CE_loss(A, Y):
    return -1*np.sum(np.log(A[Y, range(A.shape[1])]))/Y.size


#关于dW2与dW1的求导，这里用的只是便捷的近似值，精确值如何计算可看文件夹’dW矩阵求导‘
def backward_prop(Z1, A1, A2, W2, X, Y):
    dZ2 = (A2 - one_hot(Y)) / Y.size#根据CE_loss(A, Y)交叉熵损失函数求导得来
    dW2 = dZ2.dot(A1.T)
    db2 = np.mean(dZ2, axis=1).reshape(10,1)#这里一定要reshape，不然train_update_params中b1 = b1 - alpha * db1与b2 = b2 - alpha * db2后，b1与b2的形状会变成(10，10)那就不对了
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = dZ1.dot(X.T)
    db1 = np.mean(dZ1, axis=1).reshape(10,1)#同上，之所以用np.mean是因为要求10个数的概率的平均偏置值，这样才是最准确的，axis=1才是沿着正确的轴求值
    return dW1, db1, dW2, db2

def train_update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def Predictive_labels(A2):
    return np.argmax(A2, 0)

def prediction_accuracy(predictions, targe):
    return np.sum(predictions == targe) / targe.size

def train_gradient_descent(X, Y, alpha, epochs):#梯度下降
    W1, b1, W2, b2 = init_params()#参数随机初始化
    history = {'loss': [], 'accuracy': []}  # 创建一个字典来保存损失和准确率的历史记录
    for i in range(1,epochs+1):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)#前向传播
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, A2, W2, X, Y)#反向传播
        W1, b1, W2, b2 = train_update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)#更新参数

        if i % 10 == 0:#每十轮一打印
            print("epoch: ", i)#已训练轮数
            loss = CE_loss(A2, Y)  # 交叉熵损失
            print('loss：', '{:.2f}'.format(loss))
            predictions = Predictive_labels(A2)  # 做出预测
            accuracy = prediction_accuracy(predictions, Y)  # 预测准确度
            print('{:.2f}%'.format(100 * accuracy))
            history['loss'].append(loss)
            history['accuracy'].append(accuracy)
    return W1, b1, W2, b2,history

def plot_loss(history):
    epochs = range(0, len(history['loss']) * 10, 10)  # 设置 Epoch，间隔为 10
    plt.plot(epochs, history['loss'], label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    x_ticks = np.arange(0, len(history['loss']) * 10, len(history['loss']))
    floss=max(history['loss'][0:10])
    y_ticks = np.arange(0, floss, floss/10)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.legend()
    plt.savefig('./output_picture/loss.png')
    plt.show()

def plot_accuracy(history):
    epochs = range(0, len(history['accuracy']) * 10, 10)  # 设置 Epoch，间隔为 10
    plt.plot(epochs, history['accuracy'], label='Training Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    x_ticks = np.arange(0, len(history['accuracy']) * 10, len(history['accuracy']))
    lowest_accuracy=min(history['accuracy'][:10])
    y_ticks = np.arange(lowest_accuracy, 1, history['accuracy'][-1]/10)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.legend()
    plt.savefig('./output_picture/accuracy.png')
    plt.show()

def plot_sample_images(X, y):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[:, i].reshape(28, 28), cmap='gray')
        ax.set_title("Label: {}".format(y[i]))
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('./output_picture/sample_images.png')
    plt.show()

def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('./output_picture/confusion_matrix.png')
    plt.show()


# 加载和预处理数据
data = load_data('./Handwritten Digit Recognition/data.csv')
split_count = round(len(data)*0.3)#分给测试集的样本数，剩下则给训练集
train_data, test_data = split_data(data,split_count)
train_target, train_features = preprocess_data(train_data)
test_target, test_features = preprocess_data(test_data)

#训练model
alpha=0.3     #学习率
epochs=5000      #训练轮数

W1_trained, b1_trained, W2_trained, b2_trained,history_train = train_gradient_descent(train_features, train_target, alpha, epochs)

# 绘制训练过程中的损失曲线和准确率曲线
plot_loss(history_train)
plot_accuracy(history_train)


# 代入参数在test集上做预测，并且得到test集的预测准确度
def test_set_prediction(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = Predictive_labels(A2)
    return predictions

test_predictions = test_set_prediction(test_features, W1_trained, b1_trained, W2_trained, b2_trained)
test_set_accuracy=prediction_accuracy(test_predictions, test_target)
print('在测试集上预测的准确度为：','{:.2f}%'.format(100*test_set_accuracy))

# 绘制test样本图像与label
plot_sample_images(test_features, test_target)

# 计算混淆矩阵，循环遍历真实标签和预测标签，对混淆矩阵中相应的位置加一。比如，如果某个样本的真实标签是 3，模型预测的标签是 5，则在混淆矩阵的 (3, 5) 位置加一。
confusion_matrix = np.zeros((10, 10))
for true_label, predicted_label in zip(test_target, test_predictions):
    confusion_matrix[true_label, predicted_label] += 1

# 绘制混淆矩阵
plot_confusion_matrix(confusion_matrix, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
