#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>

using namespace std;

vector<vector<double>> transpose(const vector<vector<double>>& matrix) {
    int m = matrix.size();
    int n = matrix[0].size();
    vector<vector<double>> transposed(n, vector<double>(m));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            transposed[j][i] = matrix[i][j];
        }
    }
    return transposed;
}

vector<vector<int>> load_data(const string& filepath) {
    vector<vector<int>> data;
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "打开文件错误: " << filepath << endl;
        exit(1);
    }
    string line;
    getline(file, line); // 跳过第一行，因为是标签
    while (getline(file, line)) {
        stringstream ss(line);
        vector<int> row;
        string cell;
        while (getline(ss, cell, ',')) {
            row.push_back(stoi(cell));
        }
        data.push_back(row);
    }
    file.close();

    random_device rd;
    mt19937 g(rd());
    shuffle(data.begin(), data.end(), g);

    return data;
}

struct ListData {
    vector<vector<vector<double>>> dataArrays;
    ListData() {}
};

ListData split_data(const vector<vector<int>>& data, int splitCount) {
    vector<vector<double>> test(splitCount, vector<double>(data[0].size()));
    vector<vector<double>> train(data.size() - splitCount, vector<double>(data[0].size()));

    for (int i = 0; i < splitCount; i++) {
        for (size_t j = 0; j < data[0].size(); j++) {
            test[i][j] = static_cast<double>(data[i][j]);
        }
    }

    for (size_t i = splitCount; i < data.size(); i++) {
        for (size_t j = 0; j < data[0].size(); j++) {
            train[i - splitCount][j] = static_cast<double>(data[i][j]);
        }
    }
    train = transpose(train);
    test = transpose(test);

    ListData reData;
    reData.dataArrays.push_back(train);
    reData.dataArrays.push_back(test);
    return reData;
}

template<typename T, typename U>
struct PredataSaveType {
    T first;
    U second;
    PredataSaveType(T first, U second) : first(first), second(second) {}
};

PredataSaveType<vector<vector<double>>, vector<int>> preprocess_data(const vector<vector<double>>& data) {
    int target_size = data[0].size(); // 目标数组的大小与数据集中的第一个样本的特征数量相同
    vector<int> target; // 使用动态大小的向量来存储目标值
    vector<vector<double>> features(data.size() - 1, vector<double>(target_size));

    // 从数据集中提取目标值
    for (size_t j = 0; j < target_size; j++) {
        target.push_back(static_cast<int>(data[0][j]));
    }

    // 从数据集中提取特征数据
    for (size_t i = 1; i < data.size(); i++) {
        for (size_t j = 0; j < target_size; j++) {
            features[i - 1][j] = data[i][j] / 255.0;
        }
    }

    return PredataSaveType<vector<vector<double>>, vector<int>>(features, target);
}

ListData init_params() {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-0.5, 0.5);

    vector<vector<double>> W1(10, vector<double>(784));
    vector<vector<double>> b1(10, vector<double>(1));
    vector<vector<double>> W2(10, vector<double>(10));
    vector<vector<double>> b2(10, vector<double>(1));

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 784; j++) {
            W1[i][j] = dis(gen);
        }
        b1[i][0] = dis(gen);
        for (int j = 0; j < 10; j++) {
            W2[i][j] = dis(gen);
        }
        b2[i][0] = dis(gen);
    }

    ListData reData;
    reData.dataArrays.push_back(W1);
    reData.dataArrays.push_back(b1);
    reData.dataArrays.push_back(W2);
    reData.dataArrays.push_back(b2);
    return reData;
}

vector<vector<double>> ReLU(const vector<vector<double>>& Z) {
    vector<vector<double>> result(Z.size(), vector<double>(Z[0].size()));
    for (size_t i = 0; i < Z.size(); i++) {
        for (size_t j = 0; j < Z[i].size(); j++) {
            result[i][j] = max(0.0, Z[i][j]);
        }
    }
    return result;
}

vector<vector<double>> softmax(const vector<vector<double>>& Z) {
    vector<vector<double>> exp_Z(Z.size(), vector<double>(Z[0].size()));
    vector<double> zz(Z[0].size(), 0.0);
    vector<vector<double>> softmax(Z.size(), vector<double>(Z[0].size()));

    for (size_t i = 0; i < Z.size(); i++) {
        for (size_t j = 0; j < Z[i].size(); j++) {
            exp_Z[i][j] = exp(Z[i][j]);
        }
    }

    for (size_t j = 0; j < Z[0].size(); j++) {
        for (size_t i = 0; i < Z.size(); i++) {
            zz[j] += exp_Z[i][j];
        }
    }

    for (size_t i = 0; i < Z.size(); i++) {
        for (size_t j = 0; j < Z[i].size(); j++) {
            softmax[i][j] = exp_Z[i][j] / zz[j];
        }
    }

    return softmax;
}


vector<vector<double>> matrixMul(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int m = A.size();
    int n = B[0].size();
    int o = A[0].size();
    vector<vector<double>> C(m, vector<double>(n));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < o; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}


vector<vector<double>> matrixAdd(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int mA = A.size();
    int nA = A[0].size();
    int mB = B.size();
    int nB = B[0].size();

    if (mA != mB) {
        cerr << "矩阵维度不匹配，无法用广播机制相加。" << endl;
        exit(1);
    }

    vector<vector<double>> C(mA, vector<double>(nA));
    for (int i = 0; i < mA; i++) {
        for (int j = 0; j < nA; j++) {
            C[i][j] = A[i][j] + B[i][0];
        }
    }
    return C;
}


ListData forward_prop(const vector<vector<double>>& W1, const vector<vector<double>>& b1,
                      const vector<vector<double>>& W2, const vector<vector<double>>& b2,
                      const vector<vector<double>>& X) {
    vector<vector<double>> Z1, A1, Z2, A2;
    Z1 = matrixAdd(matrixMul(W1, X), b1);
    A1 = ReLU(Z1);
    Z2 = matrixAdd(matrixMul(W2, A1), b2);
    A2 = softmax(Z2);

    ListData reData;
    reData.dataArrays.push_back(Z1);
    reData.dataArrays.push_back(A1);
    reData.dataArrays.push_back(Z2);
    reData.dataArrays.push_back(A2);
    return reData;
}


vector<vector<double>> ReLU_deriv(const vector<vector<double>>& Z) {
    vector<vector<double>> result(Z.size(), vector<double>(Z[0].size()));
    for (size_t i = 0; i < Z.size(); i++) {
        for (size_t j = 0; j < Z[i].size(); j++) {
            result[i][j] = Z[i][j] > 0 ? 1 : 0;
        }
    }
    return result;
}


vector<vector<double>> one_hot(const vector<int>& Y) {
    int classes = *max_element(Y.begin(), Y.end()) + 1;
    vector<vector<double>> oneHot_Y(Y.size(), vector<double>(classes, 0.0));

    for (size_t i = 0; i < Y.size(); i++) {
        int classIndex = Y[i];
        oneHot_Y[i][classIndex] = 1.0;
    }

    return transpose(oneHot_Y);
}


double CE_loss(const vector<vector<double>>& A, const vector<int>& Y) {
    double sum = 0.0;
    int m = Y.size(); 
    
    for (int i = 0; i < m; i++) {
        int label = Y[i]; 
        sum += log(A[label][i]); 
    }

    return -sum / m; 
}


vector<vector<double>> matrixScalar(const vector<vector<double>>& matrix, double scalar) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<vector<double>> result(rows, vector<double>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = matrix[i][j] * scalar;
        }
    }
    return result;
}


vector<vector<double>> mean(const vector<vector<double>>& matrix, int axis) {
    vector<vector<double>> result;
    if (axis == 1) {
        result.resize(matrix.size(), vector<double>(1));
        for (size_t i = 0; i < matrix.size(); i++) {
            double sum = 0;
            for (size_t j = 0; j < matrix[i].size(); j++) {
                sum += matrix[i][j];
            }
            result[i][0] = sum / matrix[i].size();
        }
    } else {
        result.resize(1, vector<double>(matrix[0].size()));
        for (size_t j = 0; j < matrix[0].size(); j++) {
            double sum = 0;
            for (size_t i = 0; i < matrix.size(); i++) {
                sum += matrix[i][j];
            }
            result[0][j] = sum / matrix.size();
        }
    }
    return result;
}


vector<vector<double>> matrixSub(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int m = A.size();
    int n = A[0].size();
    vector<vector<double>> C(m, vector<double>(n));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}


vector<vector<double>> matrixElementWise(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int m = A.size();
    int n = A[0].size();
    vector<vector<double>> C(m, vector<double>(n));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = A[i][j] * B[i][j];
        }
    }
    return C;
}


ListData backward_prop(const vector<vector<double>>& Z1, const vector<vector<double>>& A1,
                       const vector<vector<double>>& A2, const vector<vector<double>>& W2,
                       const vector<vector<double>>& X, const vector<int>& Y) {
    vector<vector<double>> dZ2 = matrixScalar(matrixSub(A2, one_hot(Y)), 1.0 / Y.size());
    vector<vector<double>> dW2 = matrixMul(dZ2, transpose(A1));
    vector<vector<double>> db2 = mean(dZ2, 1);
    vector<vector<double>> dZ1 = matrixMul(transpose(W2), dZ2);
    dZ1 = matrixElementWise(dZ1, ReLU_deriv(Z1));
    vector<vector<double>> dW1 = matrixMul(dZ1, transpose(X));
    vector<vector<double>> db1 = mean(dZ1, 1);

    ListData reData;
    reData.dataArrays.push_back(dW1);
    reData.dataArrays.push_back(db1);
    reData.dataArrays.push_back(dW2);
    reData.dataArrays.push_back(db2);
    return reData;
}


vector<double> predictiveLabels(const vector<vector<double>>& A2) {
    vector<double> labels(A2[0].size());
    for (size_t j = 0; j < A2[0].size(); j++) {
        double maxVal = A2[0][j];
        int maxIndex = 0;
        for (size_t i = 1; i < A2.size(); i++) {
            if (A2[i][j] > maxVal) {
                maxVal = A2[i][j];
                maxIndex = i;
            }
        }
        labels[j] = maxIndex;
    }
    return labels;
}

double prediction_accuracy(const vector<double>& predictions, const vector<int>& target) {
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); i++) {
        if (predictions[i] == target[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / predictions.size();
}

vector<vector<double>> train_update_params(const vector<vector<double>>& params, double alpha,
                                           const vector<vector<double>>& gradients) {
    vector<vector<double>> updatedParams = params;
    for (size_t i = 0; i < params.size(); i++) {
        for (size_t j = 0; j < params[i].size(); j++) {
            updatedParams[i][j] -= alpha * gradients[i][j];
        }
    }
    return updatedParams;
}

ListData train_gradient_descent(const vector<vector<double>>& X, const vector<int>& Y, double alpha, int epochs) {
    ListData params = init_params();
    vector<vector<double>> W1 = params.dataArrays[0];
    vector<vector<double>> b1 = params.dataArrays[1];
    vector<vector<double>> W2 = params.dataArrays[2];
    vector<vector<double>> b2 = params.dataArrays[3];

    for (int i = 1; i <= epochs; i++) {
        ListData forward_result = forward_prop(W1, b1, W2, b2, X);
        vector<vector<double>> Z1 = forward_result.dataArrays[0];
        vector<vector<double>> A1 = forward_result.dataArrays[1];
        vector<vector<double>> A2 = forward_result.dataArrays[3];
        vector<double> predictions = predictiveLabels(A2);
        double loss = CE_loss(A2, Y);
        double accuracy = prediction_accuracy(predictions, Y);
        ListData backward_result = backward_prop(Z1, A1, A2, W2, X, Y);
        vector<vector<double>> dW1 = backward_result.dataArrays[0];
        vector<vector<double>> db1 = backward_result.dataArrays[1];
        vector<vector<double>> dW2 = backward_result.dataArrays[2];
        vector<vector<double>> db2 = backward_result.dataArrays[3];
        W1 = train_update_params(W1, alpha, dW1);
        b1 = train_update_params(b1, alpha, db1);
        W2 = train_update_params(W2, alpha, dW2);
        b2 = train_update_params(b2, alpha, db2);
        cout << "epoch: " << i << endl;
        cout << "loss: " << loss << endl;
        cout << fixed << setprecision(2) << (accuracy * 100) << "%" << endl;
    }

    ListData reData;
    reData.dataArrays.push_back(W1);
    reData.dataArrays.push_back(b1);
    reData.dataArrays.push_back(W2);
    reData.dataArrays.push_back(b2);
    return reData;
}


vector<double> test_set_prediction(const vector<vector<double>>& W1, const vector<vector<double>>& b1,
                                   const vector<vector<double>>& W2, const vector<vector<double>>& b2,
                                   const vector<vector<double>>& X) {
    ListData forward_result = forward_prop(W1, b1, W2, b2, X);
    vector<vector<double>> A2 = forward_result.dataArrays[3];
    vector<double> predictions = predictiveLabels(A2);
    return predictions;
}

int main() {
    string filepath = "data.csv";
    vector<vector<int>> data = load_data(filepath);
    int splitCount = round(data.size() * 0.3);
    ListData splitData = split_data(data, splitCount);
    vector<vector<double>> train_data = splitData.dataArrays[0];
    vector<vector<double>> test_data = splitData.dataArrays[1];

    PredataSaveType<vector<vector<double>>, vector<int>> train_params = preprocess_data(train_data);
    PredataSaveType<vector<vector<double>>, vector<int>> test_params = preprocess_data(test_data);
    vector<vector<double>> train_features = train_params.first;
    vector<int> train_target = train_params.second;
    vector<vector<double>> test_features = test_params.first;
    vector<int> test_target = test_params.second;

    double learning_rate = 0.3;
    int num_epochs = 100;

    ListData trained_params = train_gradient_descent(train_features, train_target, learning_rate, num_epochs);
    vector<vector<double>> W1_trained = trained_params.dataArrays[0];
    vector<vector<double>> b1_trained = trained_params.dataArrays[1];
    vector<vector<double>> W2_trained = trained_params.dataArrays[2];
    vector<vector<double>> b2_trained = trained_params.dataArrays[3];

    vector<double> predictions = test_set_prediction(W1_trained, b1_trained, W2_trained, b2_trained, test_features);
    double accuracy = prediction_accuracy(predictions, test_target);
    cout << "在测试集上预测的准确度为：" << fixed << setprecision(2) << (accuracy * 100) << "%" << endl;
}


    