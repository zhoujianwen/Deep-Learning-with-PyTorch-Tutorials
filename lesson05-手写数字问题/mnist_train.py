# -*- coding: utf-8 -*-
# @Time    : 2020/3/9 23:33
# @Author  : zhoujianwen
# @Email   : zhou_jianwen@qq.com
# @File    : mnist_train.py
# @Describe: 回顾神经网络分类任务的整体流程

import torch
from torch import nn  # 神经网络库
from torch.nn import functional as F  # 常用函数
from torch import optim #  优化工具包

import torchvision  # 视觉工具包
from matplotlib import pyplot as plt  # 数据可示化工具包

from utils import plot_image, plot_curve, one_hot

# 解决Pycharm导入模块时提示“Unresolved reference”
# 在pycharm中设置source路径
# file–>setting–>project:server–>project structure-->选择python(工程名)-->点击Sources图标-->Apply即可
# 将放package的文件夹设置为source，这样import的模块类等，就是通过这些source文件夹作为根路径来查找，也就是
# 在这些source文件夹中查找import的东西。

batch_size = 512

# step1. load dataset
# Normalize 零—均值规范化也叫标准差标准化，mean：0.1307，std：0.3081，其转化公式s = (x - mean)/std,
# 特征标准化不会改变特征取值分布，只是为了保证参数变量的取值范围具有相似的尺度，以帮助梯度下降算法收敛更快。
# shuffle 将数据集随机打乱

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

x, y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'image sample')



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # xw+b , 其中256,64的数值都是由经验决定的，28*28输入的维度，10是一个分类值0-9
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # x: [b, 1, 28, 28]
        # h1 = relu(xw1+b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2+b2)
        x = F.relu(self.fc2(x))
        # h3 = h2w3+b3
        x = self.fc3(x)
        return x



net = Net()
# [w1, b1, w2, b2, w3, b3]，optimizer是一个优化器，更新参数值
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


train_loss = []
'''
train_loader有训练样本60k，batchsize = 512，iteration = 60k/512 = 117.1875，epoch = 1
'''
for epoch in range(60):

    for batch_idx, (x, y) in enumerate(train_loader):

        # x: [b, 1, 28, 28], y: [512]
        # [b, 1, 28, 28] => [b, 784]，其中b是batchsize，28*28 => 784 可以看作是x_i的样本数据
        x = x.view(x.size(0), 28*28)
        # => [b, 10]
        out = net(x)
        # [b, 10]
        y_onehot = one_hot(y)
        # loss = mse(out, y_onehot)
        loss = F.mse_loss(out, y_onehot)  # 获得代价函数的初始值

        optimizer.zero_grad()  # 在BP之前首先将梯度清零，以保证每次更新的负梯度值是最新的。
        loss.backward()  # 计算出梯度信息
        # w' = w - lr*grad
        optimizer.step()  # 更新参数信息

        train_loss.append(loss.item())  # 保存当前参数信息

        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())  # 每训练完一个mini-batch就显示当前训练模型的参数状态

plot_curve(train_loss)  # 模型训练完毕，显示代价函数曲线收敛的走势
# we get optimal [w1, b1, w2, b2, w3, b3]  # 模型训练完之后会得到这一组最优参数解，使得loss值全局最小。

"""
这里的loss值不是用来衡量模型的性能指标，只是用来辅助我们更好地训练模型，衡量模型的性能指标有很多种方法，最终
衡量模型的指标是它的准确度。
下面使用测试集对模型进行准确度的测试。
"""

total_correct = 0
for x,y in test_loader:
    x  = x.view(x.size(0), 28*28)
    out = net(x) # 输入测试样本数据x_i，预测出概率模型
    # out: [b, 10] => pred: [b]  , 比如输出标签对应的预测概率为[0.1,0.9,0.01,......,0.08],∑P(y|x) = 1
    # argmax获得预测概率最大元素所在的索引号，max=0.9,argmax(out)=[0,1,0,......,0]，
    # 从而获得one-hot的预测编码
    # 若预测概率是out = [0.01,0.02,0.03,0.705,...,0.09],则 argmax(out) = [0,0,0,3,0,0,0,0,0,0]，
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc:', acc)

x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28*28))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')





