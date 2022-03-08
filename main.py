import numpy as np
import scipy.sparse as sp
from Networks import Networks
import matplotlib.pyplot as plt
from sklearn import manifold

# 两种PQC都需要的用到的参数
n_qubit = 4 #量子比特数
n_class = 7
layer = 1 # Block层数
nums = 140 # 训练数据数量
batch = 1
epochs = 200 # 迭代次数
prop = 0 # 残差项比例
init = np.load('data/middle_layer.npy')
label = np.load('data/labels.npy')
adj = sp.load_npz('data/adj.npz') # adj已经完成归一化了，每行和为1
net = Networks(n_qubit=n_qubit, n_class=n_class, nums=nums, label=label)
residuals = net.get_residuals(prop)

adj += residuals
init = adj @ init
init = net.normalize(init) # 归一化
train_set = init[0:nums]
test_set = init[nums:]
theta = np.random.random(layer * 8)
theta,loss,acc,test_acc = net.train1(theta, init, label[0:nums], epochs,lr=1)
np.save('data/loss'+str(prop)+'residuals.npy',loss) # 长度为epochs的向量
np.save('data/train_acc_'+str(prop)+'residuals.npy',acc) # 长度为epochs的向量
np.save('data/test_acc_'+str(prop)+'residuals.npy',test_acc)# 长度为epochs的向量


