import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.ArrayList;
import java.util.List;


public class HandwrittenDigitRecognition {

    public static double[][] transpose(double[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        double[][] transposed = new double[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }

    public static int[][] load_data(String filepath) throws IOException {
        List<int[]> dataList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filepath))) {
            br.readLine();//跳过第一行，因为是标签
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(","); 
                int[] row = new int[values.length];
                for (int i = 0; i < values.length; i++) {
                    row[i] = Integer.parseInt(values[i]);
                }
                dataList.add(row);
            }
        }

        // 变成二维数组
        int[][] data = new int[dataList.size()][];
        for (int i = 0; i < dataList.size(); i++) {
            data[i] = dataList.get(i);
        }

        // 随机打乱
        Random rnd = new Random();
        for (int i = data.length - 1; i > 0; i--) {
            int index = rnd.nextInt(i + 1);
            // Simple swap
            int[] a = data[index];
            data[index] = data[i];
            data[i] = a;
        }

        return data;
    }
 
    public static class ListData {
        public List<double[][]> dataArrays;//dataArrays 是一个 List，它存储的元素是 double[][] 类型的二维数组。
        public ListData() {
            this.dataArrays = new ArrayList<>();
        }
    }

    public static ListData split_data(int[][] data, int splitCount) {
        double[][] test = new double[splitCount][data[0].length];
        double[][] train = new double[data.length - splitCount][data[0].length];
        
        // 将test部分转换为double
        for (int i = 0; i < splitCount; i++) {
            for (int j = 0; j < data[0].length; j++) {
                test[i][j] = (double) data[i][j];
            }
        }
        
        // 将train部分转换为double
        for (int i = splitCount; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                train[i - splitCount][j] = (double) data[i][j];
            }
        }
        train=transpose(train);
        test=transpose(test);

        ListData reData = new ListData();
        reData.dataArrays.add(train);
        reData.dataArrays.add(test);
        return reData;
    }
    
    public static class PredataSaveType<T, U> {
        public final T first;
        public final U second;
    
        public PredataSaveType(T first, U second) {
            this.first = first;
            this.second = second;
        }
    }

    public static List<PredataSaveType<double[][],int[]>> preprocess_data(double[][] data) {
        int[] target = new int[data[0].length];
        double[][] features = new double[data.length - 1][data[0].length];

        for (int j = 0; j < data[0].length; j++) {
            target[j] = (int)data[0][j];
            for (int i = 1; i < data.length; i++) {
                features[i-1][j] = data[i][j] / 255.0;
            }
        }
        
        List<PredataSaveType<double[][],int[]>> dataList = new ArrayList<>();
        dataList.add(new PredataSaveType<>(features,target));
        return dataList;
    }

    public static ListData init_params() {
        Random rand = new Random();
        double[][] W1 = new double[10][784];
        double[][] b1 = new double[10][1];
        double[][] W2 = new double[10][10];
        double[][] b2 = new double[10][1];
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 784; j++) {
                W1[i][j] = rand.nextDouble() - 0.5;
            }
            b1[i][0] = rand.nextDouble() - 0.5;
            for (int j = 0; j < 10; j++) {
                W2[i][j] = rand.nextDouble() - 0.5;
            }
            b2[i][0] = rand.nextDouble() - 0.5;
        }
        ListData reData = new ListData();
        reData.dataArrays.add(W1);
        reData.dataArrays.add(b1);
        reData.dataArrays.add(W2);
        reData.dataArrays.add(b2);
        return reData;
    }

    public static double[][] ReLU(double[][] Z) {
        double[][] result = new double[Z.length][Z[0].length];
        for (int i = 0; i < Z.length; i++) {
            for (int j = 0; j < Z[i].length; j++) {
                result[i][j] = Math.max(0, Z[i][j]);
            }
        }
        return result;
    }

    public static double[][] softmax(double[][] Z) {

        double[][] exp_Z = new double[Z.length][Z[0].length];
        double[] zz = new double[Z[0].length];
        double[][] softmax = new double[Z.length][Z[0].length];

        // 计算exp_Z
        for (int i = 0; i < Z.length; i++) {
            for (int j = 0; j < Z[i].length; j++) {
                exp_Z[i][j] = Math.exp(Z[i][j]);
            }
        }

        // 按列相加得到zz
        for (int j = 0; j < Z[0].length; j++) {
            for (int i = 0; i < Z.length; i++) {
                zz[j] += exp_Z[i][j];
            }
        }

        // 计算softmax
        for (int i = 0; i < Z.length; i++) {
            for (int j = 0; j < Z[i].length; j++) {
                softmax[i][j] = exp_Z[i][j] / zz[j];
            }
        }

        return softmax;
    }

    public static ListData forward_prop(double[][] W1, double[][] b1, double[][] W2, double[][] b2, double[][] X) {
        double[][] Z1 = matrixAdd(matrixMul(W1, X), b1);
        double[][] A1 = ReLU(Z1);
        double[][] Z2 = matrixAdd(matrixMul(W2, A1), b2);
        double[][] A2 = softmax(Z2);
        
        ListData reData = new ListData();
        reData.dataArrays.add(Z1);
        reData.dataArrays.add(A1);
        reData.dataArrays.add(Z2);
        reData.dataArrays.add(A2);
        return reData;
    }

    public static double[][] matrixMul(double[][] A, double[][] B) {
        int m = A.length;
        int n = B[0].length;
        int o = A[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < o; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    public static double[][] matrixAdd(double[][] A, double[][] B) {
        int mA = A.length;
        int nA = A[0].length;
        int mB = B.length;
        int nB = B[0].length;
    
        // 检查是否能用广播机制，涉及矩阵相加的只有偏置项b
        if (mA != mB) {
            System.err.printf("mA: %d, nA: %d, mB: %d, nB: %d\n", mA, nA, mB, nB);
            throw new IllegalArgumentException("矩阵维度不匹配，无法用广播机制相加");
        }
    
        double[][] C = new double[mA][nA];
        for (int i = 0; i < mA; i++) {
            for (int j = 0; j < nA; j++) {
                C[i][j] = A[i][j] + B[i][0];
            }
        }
        return C;
    } 

    public static double[][] ReLU_deriv(double[][] Z) {
        double[][] result = new double[Z.length][Z[0].length];
        for (int i = 0; i < Z.length; i++) {
            for (int j = 0; j < Z[i].length; j++) {
                result[i][j] = Z[i][j] > 0 ? 1 : 0;
            }
        }
        return result;
    }

    public static double[][] one_hot(int[] Y) {
        // 计算类别数量
        int classes = Arrays.stream(Y).max().getAsInt() + 1;

        // 创建一个全零矩阵
        double[][] oneHot_Y = new double[Y.length][classes];

        // 对每个样本的类别进行独热编码
        for (int i = 0; i < Y.length; i++) {
            int classIndex = Y[i];
            oneHot_Y[i][classIndex] = 1.0;
        }
        // 转置矩阵，将每个类别的独热编码表示为一个列向量
        return transpose(oneHot_Y);
    }

    public static double CE_loss(double[][] A, int[] Y) {
        double sum = 0.0;
        int m = Y.length; // label数量
        
        // 计算交叉熵损失
        for (int i = 0; i < m; i++) {
            int label = Y[i]; // 当前样本的标签
            sum += Math.log(A[label][i]); // 对应类别的概率损失
        }

        // 取平均
        return -sum / m;
    }

    public static double[][] matrixScalar(double[][] matrix, double scalar) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = matrix[i][j] * scalar;
            }
        }
        return result;
    }

    public static ListData backward_prop(double[][] Z1, double[][] A1, double[][] A2, double[][] W2, double[][] X, int[] Y) {
        double[][] dZ2 = matrixScalar(matrixSub(A2, one_hot(Y)), 1.0 / Y.length);
        double[][] dW2 = matrixMul(dZ2, transpose(A1));
        double[][] db2 = mean(dZ2, 1);
        double[][] dZ1 = matrixMul(transpose(W2), dZ2);
        dZ1 = matrixElementWise(dZ1, ReLU_deriv(Z1));
        double[][] dW1 = matrixMul(dZ1, transpose(X));
        double[][] db1 = mean(dZ1, 1);

        ListData reData = new ListData();
        reData.dataArrays.add(dW1);
        reData.dataArrays.add(db1);
        reData.dataArrays.add(dW2);
        reData.dataArrays.add(db2);
        return reData;
    }

    public static double[][] mean(double[][] matrix, int axis) {
        double[][] result;
        if (axis == 1) {
            // 对每行求均值
            result = new double[matrix.length][1];
            for (int i = 0; i < matrix.length; i++) {
                double sum = 0;
                for (int j = 0; j < matrix[i].length; j++) {
                    sum += matrix[i][j];
                }
                result[i][0] = sum / matrix[i].length;
            }
        } else {
            // 对每列求均值
            result = new double[1][matrix[0].length];
            for (int j = 0; j < matrix[0].length; j++) {
                double sum = 0;
                for (int i = 0; i < matrix.length; i++) {
                    sum += matrix[i][j];
                }
                result[0][j] = sum / matrix.length;
            }
        }
        return result;
    }

    public static double[][] matrixSub(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] - B[i][j];
            }
        }
        return C;
    }

    public static double[][] matrixElementWise(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] * B[i][j];
            }
        }
        return C;
    }

    public static double[] predictiveLabels(double[][] A2) {
        double[] labels = new double[A2[0].length]; // 创建一个与列数相同长度的一维数组
        for (int j = 0; j < A2[0].length; j++) {
            double maxVal = A2[0][j]; // 假设第一行的元素为最大值
            int maxIndex = 0; // 最大值的索引
            // 在每一列中找到最大值及其索引
            for (int i = 1; i < A2.length; i++) {
                if (A2[i][j] > maxVal) {
                    maxVal = A2[i][j];
                    maxIndex = i;
                }
            }
            labels[j] = maxIndex; // 将最大值的索引存储在结果数组中
        }
        return labels;
    }

    public static ListData train_gradient_descent(double[][] X, int[] Y, double alpha, int epochs) {
        ListData params = init_params();
        double[][] W1 = params.dataArrays.get(0);
        double[][] b1 = params.dataArrays.get(1);
        double[][] W2 = params.dataArrays.get(2);
        double[][] b2 = params.dataArrays.get(3);

        for (int i = 1; i < epochs+1; i++) {
            ListData forward_result = forward_prop(W1, b1, W2, b2, X);
            double[][] Z1 = forward_result.dataArrays.get(0);
            double[][] A1 = forward_result.dataArrays.get(1);
            double[][] A2 = forward_result.dataArrays.get(3);
            double[] predictions = predictiveLabels(A2);
            double loss = CE_loss(A2, Y);
            double accuracy = prediction_accuracy(predictions, Y);
            ListData backward_result = backward_prop(Z1, A1, A2, W2, X, Y);
            double[][] dW1 = backward_result.dataArrays.get(0);
            double[][] db1 = backward_result.dataArrays.get(1);
            double[][] dW2 = backward_result.dataArrays.get(2);
            double[][] db2 = backward_result.dataArrays.get(3);
            W1 = train_update_params(W1, alpha, dW1);
            b1 = train_update_params(b1, alpha, db1);
            W2 = train_update_params(W2, alpha, dW2);
            b2 = train_update_params(b2, alpha, db2);
            System.out.println("epoch: " + i);
            System.out.println("loss: " + loss);
            System.out.println(String.format("%.2f%%", accuracy * 100));
            
        }
        ListData reData = new ListData();
        reData.dataArrays.add(W1);
        reData.dataArrays.add(b1);
        reData.dataArrays.add(W2);
        reData.dataArrays.add(b2);
        return reData;
    }

    public static double prediction_accuracy(double[] predictions, int[] target) {
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == target[i]) {
                correct++;
            }
        }
        return (double) correct / predictions.length;
    }

    public static double[][] train_update_params(double[][] params, double alpha, double[][] gradients) {
        int rows = params.length;
        int cols = params[0].length;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                params[i][j] -= alpha * gradients[i][j];
            }
        }
        return params;
    }

    public static double[] test_set_prediction(double[][] W1, double[][] b1, double[][] W2, double[][] b2, double[][] X) {
        ListData forward_result = forward_prop(W1, b1, W2, b2, X);
            double[][] A2 = forward_result.dataArrays.get(3);
            double[] predictions = predictiveLabels(A2);
            return predictions;
    }
    
    public static void main(String[] args) throws IOException {
        String filepath = "src/Handwritten Digit Recognition/data.csv";
        int[][] data = load_data(filepath);
        int splitCount = (int) Math.round(data.length * 0.3);
        ListData splitData = split_data(data, splitCount);
        double[][] train_data = splitData.dataArrays.get(0);
        double[][] test_data = splitData.dataArrays.get(1);
    
        List<PredataSaveType<double[][], int[]>> PreprocessedTrainData = preprocess_data(train_data);
        double[][] train_features = PreprocessedTrainData.get(0).first;
        int[] train_target = PreprocessedTrainData.get(0).second;

        double alpha = 0.3;
        int epochs = 100;
        ListData W_trained = train_gradient_descent(train_features, train_target, alpha, epochs);

        double[][] W1_trained = W_trained.dataArrays.get(0);
        double[][] b1_trained = W_trained.dataArrays.get(1);
        double[][] W2_trained = W_trained.dataArrays.get(2);
        double[][] b2_trained = W_trained.dataArrays.get(3);

        List<PredataSaveType<double[][], int[]>> PreprocessedTestData = preprocess_data(test_data);
        double[][] test_features = PreprocessedTestData.get(0).first;
        int[] test_target = PreprocessedTestData.get(0).second;
        
        double[] predictions = test_set_prediction(W1_trained, b1_trained, W2_trained, b2_trained, test_features);
        double accuracy = prediction_accuracy(predictions, test_target);
        System.out.println("在测试集上预测的准确度为："+String.format("%.2f%%", accuracy * 100));
    }
}