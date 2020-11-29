# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:34:47 2020
BP算法
定义神经网络network类:参数输入数组规定每层的神经元个数：包括输入层、隐藏层和输出层
其中的方法如下：
__init__:初始化神经网络的结构以及初始权重和偏置
backprop: 实现前向、后向计算
evaluate: 参数评估函数，测试模型预测正确率
update_mini_batch: 批量随机梯度函数
SGD:SGD训练网络函数
@author: 地三仙
坑：
self.update_mini_batch(self,batch, eta) # !!!!!!!!!类里面调用函数 参数不能加self,默认有!!!!!!!!
重点是搞清楚矩阵运算：dot(w,a) + b 观察shape是否一致 注意矩阵和向量与矩阵和矩阵(,1)的区别
# len(train_data) TypeError: object of type 'zip' has no len()  line 103 zip是迭代器
构建训练数据集 list(zip(x_train, y_train)) 保证可以k,v遍历 且可以len() 
# 根据小批量batch进行更新参数 用切片[:] 直接索引会改变层级结构
"""

import numpy as np

class Network(object):
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.size = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]] # y,1 是矩阵
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]
                        
    def backprop(self, x, y):
        "返回参数w,b的微分"
        # w,b微分初始值赋值为零
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x.reshape(-1,1)  # (n,1)
        activations = [activation] # 存激活函数值
        zs = [] # 存放z向量，即激活前的数值
        # 前向传递
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,activation) + b  # shape (n,1)
            zs.append(z)
            activation = self.sigmoid(z)  # todo 激活函数待定义
            activations.append(activation)
          
        # 后向传递
#        delta = self.cost_derivative(activation, y)
        delta = self.cost_derivative(activations[-1], y.reshape(-1, 1)) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer - 1].transpose()) 

        return (nabla_b, nabla_w)
        
            
    def sigmoid(self,z):
        """激活函数"""
        return 1.0 / (1 + np.exp(-z))  # z值错误赋值  RuntimeWarning: overflow encountered in exp
    
    def sigmoid_prime(self,z):
        """求导"""
        return self.sigmoid(z) * (1 - self.sigmoid(z))
        
    def cost_derivative(self,output,y):
        return (output-y)
    
    def update_mini_batch(self, batch, eta):
        """ 
        使用backprop函数，更新权重w和偏置b
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]  
        
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        self.biases = [b - (eta / len(batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - (eta / len(batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        
    def feedforward(self, a):
       """
       返回最终输出值:(n,1)
       """
       for b, w in zip(self.biases, self.weights):
           a = self.sigmoid(np.dot(w, a.reshape(-1,1)) + b) 
#           a = self.sigmoid(np.dot(w, a) + b) # 矩阵运算 右边如果是向量 最后结果成为向量 载加矩阵b乱套了
       return a

       
    def evaluate(self, test_data):
        """
        返回正确的测试数量 y是独热编码 (10,)
        """
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]  #  得到最大元素值得索引 按理应该对feedforward(x) 先转成(n,)
        
        return sum(int(x==y) for (x, y) in test_results)  # 中括号不是必须
        
                        
    def SGD(self, training_data, epochs,mini_batch_size, eta, test_data=None):
        """
        使用SGD训练网络
        样本(x, y): 输入数据为x->[x1,x2,x3,,,xm]，label为y->[y1,y2,,,yn]
        """
        if test_data: 
            n_test = len(test_data)
        
        for j in range(epochs):
            np.random.shuffle(training_data)  # todo 打散数据顺序
            mini_batches = [training_data[k: k + mini_batch_size] for 
                            k in range(0,len(training_data), mini_batch_size)]
            for batch in mini_batches:  # 根据小批量batch进行更新参数 用切片[:] 直接索引会改变层级结构
                self.update_mini_batch(batch, eta) # !!!!!!!!!类里面调用函数 参数不能加self默认有 !!!!!!!!
        
            if test_data:
                print("Epoch {0}:{1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

