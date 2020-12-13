import sys
import numpy as np
import random


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def soft_max(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Dnn:
    def __init__(self, train_x, train_y, test_x, epochs, lr, batch_size, beta, batches):
        self.__train_x = train_x
        self.__train_y = train_y
        self.__test_x = test_x
        self.__w1 = np.random.rand(128, 784) / np.sqrt(784)
        self.__b0 = np.zeros((128, 1)) / np.sqrt(784)
        self.__w2 = np.random.rand(10, 128) / np.sqrt(128)
        self.__b1 = np.zeros((10, 1)) / np.sqrt(128)
        self.__lr = lr
        self.__epochs = epochs
        self.__loss = []
        self.__batch_size = batch_size
        self.__beta = beta
        self.__batches = batches

    def train(self):
        dW1, dW2, db0, db1 = None, None, None, None
        for i in range(self.__epochs):
            print("epoch: ", i)
            permutation = np.random.permutation(self.__train_x.shape[1])
            x_train_shuffled = self.__train_x[:, permutation]
            y_train_shuffled = self.__train_y[:, permutation]
            for j in range(self.__batches):
                begin = j * self.__batch_size
                end = min(begin + self.__batch_size, self.__train_x.shape[1] - 1)
                if begin > end:
                    break
                X = x_train_shuffled[:, begin:end]
                Y = y_train_shuffled[:, begin:end]
                m_batch = end - begin
                h1, h2 = self.forward_computation(X)  # h1 = sig(W1*x_bach + b0).... h2 = soft_max(W2*h1+b1)
                # now make the back propagation for 1 hidden label:
                delta_2 = (h2 - Y)
                delta_1 = np.multiply(self.__w2.T @ delta_2, np.multiply(h1, 1 - h1))
                if i != 0:  # update the new w1 w2
                    dW1_old = dW1
                    dW2_old = dW2
                    db0_old = db0
                    db1_old = db1
                    dW1 = delta_1 @ X.T
                    dW2 = delta_2 @ h1.T
                    db0 = np.sum(delta_1, axis=1, keepdims=True)
                    db1 = np.sum(delta_2, axis=1, keepdims=True)
                    """
                    we make a beta parameter for momentum. 
                    """
                    dW1 = (self.__beta * dW1_old + (1. - self.__beta) * dW1)
                    db0 = (self.__beta * db0_old + (1. - self.__beta) * db0)
                    dW2 = (self.__beta * dW2_old + (1. - self.__beta) * dW2)
                    db1 = (self.__beta * db1_old + (1. - self.__beta) * db1)
                else:  # epoch = 0 set the new dW1, dW2, db0, db1
                    dW1 = delta_1 @ X.T
                    dW2 = delta_2 @ h1.T
                    db0 = np.sum(delta_1, axis=1, keepdims=True)
                    db1 = np.sum(delta_2, axis=1, keepdims=True)
                # update the weights vectors
                self.__w1 = self.__w1 - (1. / m_batch) * dW1 * self.__lr
                self.__b0 = self.__b0 - (1. / m_batch) * db0 * self.__lr
                self.__w2 = self.__w2 - (1. / m_batch) * dW2 * self.__lr
                self.__b1 = self.__b1 - (1. / m_batch) * db1 * self.__lr

        return self

    def forward_computation(self, x_set):
        """
        we have 1 hidden label so we dont need loop we can calculate immediately
        :param x_set: V0 - the observations
        :return: x1 =Z OF label 1. x2 = V1
        """
        h1 = sigmoid(self.__w1 @ x_set + self.__b0)
        h2 = soft_max(self.__w2 @ h1 + self.__b1)
        return h1, h2

    def predict(self):
        y_predict = open("test_y", 'w')
        x_tst_tmp = np.transpose(self.__test_x)
        for xi in x_tst_tmp:
            xi = np.expand_dims(xi, axis=1)
            x1, x2 = self.forward_computation(xi)
            y_hat = np.argmax(x2)
            y_predict.write(str(y_hat) + "\n")
        y_predict.close()


def main():
    train_x_short = np.loadtxt(sys.argv[1])
    train_y_short = np.loadtxt(sys.argv[2], dtype='int')
    test_x = np.loadtxt(sys.argv[3])
    # for training take only 5000
    train_x_short = (np.transpose(train_x_short)) / 255
    test_x = (np.transpose(test_x)) / 255
    # one hot encoding:
    train_y_short_one_hot = np.zeros((train_y_short.size, train_y_short.max() + 1))
    train_y_short_one_hot[np.arange(train_y_short.size), train_y_short] = 1
    train_y_short_one_hot = np.transpose(train_y_short_one_hot)
    Dnn(train_x_short, train_y_short_one_hot, test_x, 400, 0.1, 200, 0.9, 1000).train().predict()


if __name__ == '__main__':
    main()
