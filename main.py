# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 13:32:36 2020
利用建立好的bp模型，进行图像分类的验证
@author: 地三仙
# one-hot编码的结果是比较奇怪的，最好是先转换成二维数组
# 展平方法：a.flatten() a是np.array,函一维数组
"""
from bp import Network
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from datetime import datetime as t
import matplotlib.pyplot as plt

# 1.数据预处理
# 从本地获取数字图像的数据集 .npz
def get_data(path):
    f = np.load(path)
    x_train_all, y_train_all = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return x_train_all, y_train_all, x_test, y_test
# 1.1数据集划分
path = "C:/Users/地三仙/.keras/datasets/fashion-mnist/mnist.npz"  # 数据条目一样 npz文件比压缩文件更小
x_train_all, y_train_all, x_test, y_test = get_data(path)
x_validate, y_validate = x_train_all[ :5000], y_train_all[ :5000] # 本案例没有使用
x_train, y_train = x_train_all[5000: ], y_train_all[5000: ]
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 查看数据
def show_images(n_rows, n_cols ,x_data, y_data):
    assert len(x_data) == len(y_data)
    assert n_rows * n_cols < len(x_data)
    plt.figure(figsize = (n_cols * 1.4, n_rows * 1.6))  # 宽、高
    for row in range(n_rows):
        for col in range(n_cols):
            index = row * n_cols + col
            plt.subplot(n_rows, n_cols, index+1)
            plt.imshow(x_data[index],cmap='binary')
            plt.axis('off')
            plt.title(str(y_data[index]))
    plt.show()

show_images(5, 3 ,x_train[:100], y_train[:100])
              
               
# 1.2数据展平:三维数据变二维数据
def flatten(tri_arr):
    data = [a.flatten() for a in tri_arr]
    return np.array(data)
    
x_train_flatten = flatten(x_train)
x_validate_flatten = flatten(x_validate)
x_test_flatten = flatten(x_test)
print(x_train_flatten.shape)

# 1.3对标签数据进行独热编码
enc = OneHotEncoder()
enc.fit(y_train_all.reshape(-1,1)) 
# one-hot编码的结果是比较奇怪的，最好是先转换成二维数组
y_train_onehot = enc.transform(y_train.reshape(-1,1)).toarray()
y_validate_onehot = enc.transform(y_validate.reshape(-1,1)).toarray()
y_test_onehot = enc.transform(y_test.reshape(-1,1)).toarray()
print(x_train_flatten.shape, y_train_onehot.shape)
print(y_test_onehot[0].shape)

# 2.构建模型
net = Network([784, 100, 50, 10]) # 28X28展平为784 两层隐藏层
start = t.now()
print("开始时间：" , start)
net.SGD(list(zip(x_train_flatten, y_train_onehot)), 3, 16, 2,
        list(zip(x_test_flatten, y_test_onehot)))
end = t.now()
print("结束时间：", end)
print("训练用时：", end - start)

# 3.预测一下
def predict(net, x):
    y_predict = net.feedforward(x.flatten())
    print("预测样本为")
    plt.imshow(x, cmap='binary')
    plt.show()
    print("预测值为：" + str(np.argmax(y_predict)))

predict(net, x_validate[2])



