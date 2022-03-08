import random

from QuantumGate import QuantumGate
import numpy as np
import scipy.sparse as sp
from numpy import pi, sin, cos, sqrt, exp
from copy import deepcopy

class Networks:
    def __init__(self, n_qubit, n_class, nums, label):
        self.nums = nums
        self.label = label
        self.n_class = n_class
        self.n_qubit = n_qubit
        self.map = {0:[0,1], 1:[2,3], 2:[4,5], 3:[6,7], 4:[8,9],
                    5:[10,11], 6:[12,13]}
    def uniformstate(self,sigma,nums,init):
        psi = np.zeros((nums,2**self.n_qubit))
        for j in range(nums):
            for i in range(2**self.n_qubit):
                psi[j,i] = (1+np.random.normal(0,sigma,1)) * init[i]

        return psi


    #随机采样，默认值10
    def randomSample(self,psi,label):
        ind = random.sample(list(range(0,len(psi))),self.nums)
        return psi[ind],label[ind]

    # 归一化
    def normalize(self,psi):
        h,l = psi.shape # h是psi的个数，l是每个psi的维度
        for j in range(h):
            s = sqrt(sum(psi[j]**2))
            for i in range(l):
                psi[j,i] = psi[j,i] / s
        return psi

    def train1(self, theta, init, label, epochs, lr):  # 输入为随机采样后的量子态集合
        expect = np.zeros((epochs, self.nums, self.n_class))
        acc = np.zeros(epochs)
        test_acc = np.zeros(epochs)
        psi = init[0:self.nums]  # 训练集数据
        # y = label
        y = np.zeros((self.nums, self.n_class))
        for k in range(self.nums):
            y[k, label[k]] = 1
        loss = np.zeros(epochs)
        for epoch in range(epochs):
            delta = np.zeros_like(theta)  # 每个epoch更新一次梯度
            for i in range(self.nums):
                # 先根据现有的参数计算一次期望概率
                expect[epoch, i] = self.getExpectation1(theta, psi[i].reshape(len(psi[i]), 1))

                for t in range(len(theta)):
                    grad_e = self.getGradient1(theta, t, psi[i].reshape(len(psi[i]), 1))
                    soft_e = self.Softmax(deepcopy(expect[epoch, i]))
                    delta[t] += lr * (soft_e - y[i]).reshape((1, self.n_class)) @ grad_e.reshape((self.n_class, 1))

            theta -= delta / self.nums  # 更新参数
            tmp = 0
            for i in range(self.nums):
                tmp -= np.log(expect[epoch, i, label[i]])  # 计算损失函数
            loss[epoch] = tmp
            acc[epoch] = self.get_accuracy(expect[epoch], label)
            test_acc[epoch] = self.test(theta=theta, init=init)
            print('第', epoch, '次迭代，', 'loss:', loss[epoch], 'train_acc:', acc[epoch],
                  'test_acc', test_acc[epoch])

        return theta, loss, acc, test_acc

    def getGradient1(self, theta, num_para, init):  # 返回PSR的结果
        left = deepcopy(theta)
        right = deepcopy(theta)
        left[num_para] = left[num_para] - pi / 4
        right[num_para] = right[num_para] + pi / 4
        # 左边
        out_l = self.getBlock1(theta=left) @ init
        expect_l = np.zeros(self.n_class)
        for i in range(self.n_class):
            expect_l[i] = out_l[self.map[i][0], 0] ** 2 + out_l[self.map[i][1], 0] ** 2
        # 右边
        out_r = self.getBlock1(theta=right) @ init
        expect_r = np.zeros(self.n_class)
        for i in range(self.n_class):
            expect_r[i] = out_r[self.map[i][0], 0] ** 2 + out_r[self.map[i][1], 0] ** 2

        return expect_r - expect_l

    def getExpectation1(self, theta, init):
        res = self.getBlock1(theta=theta) @ init
        expect = np.zeros(self.n_class)
        for i in range(self.n_class):
            expect[i] = res[self.map[i][0], 0] ** 2 + res[self.map[i][1], 0] ** 2
        return expect

    def getBlock1(self, theta):  # 常见的PQC block
        QG = QuantumGate()
        layer = len(theta) // 8
        U = np.eye(2 ** self.n_qubit)
        for lay in range(layer):
            U1 = np.kron(
                np.kron(
                    np.kron(self.Ry(theta[lay * 8 + 0]), self.Ry(theta[lay * 8 + 1])),
                    self.Ry(theta[lay * 8 + 2])),
                self.Ry(theta[lay * 8 + 3]))
            U2 = np.kron(QG.C1nU(n=0, U=self.Ry(theta=theta[lay * 8 + 4])), np.eye(4))
            U3 = np.kron(np.kron(QG.I, QG.C1nU(n=0, U=self.Ry(theta=theta[lay * 8 + 5]))), QG.I)
            U4 = np.kron(np.eye(4), QG.C1nU(n=0, U=self.Ry(theta=theta[lay * 8 + 6])))
            U5 = QG.UnC1(n=2, U=self.Ry(theta=theta[lay * 8 + 7]))
            U = U @ U5 @ U4 @ U3 @ U2 @ U1

        return U

    def Softmax(self, x):
        A = sum(np.exp(x))
        for k in range(len(x)):
            x[k] = np.exp(x[k]) / A
        return x

    def partial_NLL(self, x, y):
        return sum(x - y)

    def Ry(self, theta):
        return np.array([[cos(theta), -sin(theta)],
                         [sin(theta), cos(theta)]])

    def test(self, theta, init):
        test_expect = np.zeros((len(self.label) - self.nums, self.n_class))
        for n in range(self.nums, len(self.label)):
            test_expect[n - self.nums] = self.getExpectation1(theta=theta, init=init[n].reshape(len(init[n]), 1))
        test_acc = self.get_accuracy(test_expect, self.label[self.nums:])
        return test_acc

    def get_accuracy(self, expect, label):
        # expect的shape为：[num, n_class]
        acc = 0
        for j in range(expect.shape[0]):
            arg = np.argmax(expect[j])
            if arg == label[j]:
                acc += 1
        return acc

    def get_residuals(self, prop):
        # adj中共有13264条边，
        # 我们假设残差项矩阵不为0的元素为13264*prop，
        # 同时应保证其范数为2708*prop
        a = np.random.randint(low=0, high=2708, size=int(13264 * prop / 2))
        b = np.random.randint(low=0, high=2708, size=int(13264 * prop / 2))
        row = np.hstack((a, b))
        col = np.hstack((b, a))
        data = (np.ones_like(row) -
                2 * np.random.randint(low=0, high=2, size=len(row))) * prop

        residuals = sp.coo_matrix((data, (row, col)), shape=(2708, 2708))

        return residuals




