import numpy as np
import scipy.sparse as sp
from math import cos, sin
import random
import operator

class QuantumGate(object):
    def __init__(self):
        self.I = np.eye(2)
        self.X = np.array([[0,1],[1,0]])
        self.Y = np.array([[0,1j],[-1j,0]])
        self.Z = np.array([[1,0],[0,-1]])
        self.H = 1/np.sqrt(2)* np.array([[1,1],[1,-1]])
        self.O1 = np.array([[1, 0], [0, 0]])
        self.O2 = np.array([[0, 0], [0, 1]])

    # n表示控制比特和目标比特之间隔了n个qubit,当控制比特为1是，U作用到目标比特,高位控制低位
    def C1nU(self,n=0, U=np.array([[0,1],[1,0]])):
        a = int(np.log2(U.shape[0]))
        zero = np.kron(self.O1, np.eye(2 ** (n + a)))
        one = np.kron(self.O2, np.kron(np.eye(2 ** n), U))
        return zero + one
    # n表示控制比特和目标比特之间隔了n个qubit,当控制比特为0是，U作用到目标比特,高位控制低位
    def C0nU(self,n=0, U=np.array([[0,1],[1,0]])):
        a = int(np.log2(U.shape[0]))
        one = np.kron(self.O2,np.eye(2 ** (n + a)))
        zero = np.kron(self.O1,np.kron(np.eye(2 ** n), U))
        return zero + one
    # 高位目标比特，低位控制比特，控制比特为0，目标比特作用U
    def UnC0(self,n=0, U=np.array([[0,1],[1,0]])):
        a = int(np.log2(U.shape[0]))
        one = np.kron(np.eye(2 ** (n + a)),self.O2)
        zero = np.kron(np.kron(U, np.eye(2 ** n)),self.O1)
        return zero + one
    # 高位目标比特，低位控制比特，控制比特为1，目标比特作用U
    def UnC1(self,n=0, U=np.array([[0,1],[1,0]])):
        a = int(np.log2(U.shape[0]))
        zero = np.kron(np.eye(2 ** (n + a)),self.O1)
        one = np.kron(np.kron(U, np.eye(2 ** n)),self.O2)
        return zero + one

    def layer(self,n_qubit,gate_List):
        temp = 1
        for gate in gate_List:
            temp = np.kron(temp,gate)
        if temp.shape[0] != 2**n_qubit:
            print(len(temp))
            print('该层量子门数与量子比特数不一致')
            return None
        return temp

    def block_Z(self,n_qubit):
        ent = sp.eye(2**n_qubit)
        for i in range(n_qubit-1):
            pre = sp.eye(2 ** i)
            post = sp.eye(2 ** (n_qubit-2-i))
            ent = sp.kron(sp.kron(pre, self.C1nU(n=0,U=self.X)),post) @ ent
        ent = self.UnC1(n=n_qubit-2, U=self.X) @ ent
        return ent

    def block_CRz(self,n_qubit, phi):
        ent = np.eye(2 ** n_qubit)
        for i in range(n_qubit - 1):
            pre = np.eye(2 ** i)
            post = np.eye(2 ** (n_qubit - 2 - i))
            Rz = np.array([[1, 0], [0, phi[i]]])
            ent = np.kron(np.kron(pre, self.C1nU(n=0, U=Rz)), post) @ ent
        Rz = np.array([[1, 0], [0, phi[-1]]])
        ent = self.UnC1(n=n_qubit - 2, U=Rz) @ ent

        return ent

if __name__ == "__main__":
    n_qubit = 3
    QG = QuantumGate()
    ent = QG.block_Z(n_qubit)
    print(ent)













