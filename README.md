运行环境：win10 python 3.6.5 matplotlib 2.2.2 numpy 1.18.5 sklearn 0.23.2

基于反向传播批量梯度下降算法构建全连接神经网络，以数字验证码图像数据集进行测试。 
BP算法 定义神经网络network类:参数输入数组规定每层的神经元个数：包括输入层、隐藏层（可含多个）和输出层 
主要方法如下： init:初始化神经网络的结构以及初始权重w和偏置b 
backprop: 实现前向、后向计算 
evaluate: 参数评估函数，测试模型预测正确率 
update_mini_batch: 批量更新权重函数 
SGD:训练函数。training_data格式为(x, y): 输入数据为x->[x1,x2,x3,,,xm]，label为y->[y1,y2,,,yn] 
主程序选择mnist数字验证码图像数据集，采取本地加载npz格式文件方式。

