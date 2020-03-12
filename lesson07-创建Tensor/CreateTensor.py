#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/12 14:03
# @Author  : zhoujianwen
# @Email   : zhou_jianwen@qq.com
# @File    : CreateTensor.py
# @Describe: 创建Tensor


import numpy as np
import torch

a = np.array([2,3.3])
# 从numpy导入的float其实是double类型
print("torch from numpy:",torch.from_numpy(a))

b = np.ones([2,3]) # 创建一个二行三列的S元素全是的数组
print("torch from numpy",torch.from_numpy(b))

c = torch.tensor([2.,3.2])  # 小写的tensor只能接受现成的数据
print("c:",c)
print("c:",c.type())

d = torch.Tensor(2,3)  # 大写的tensor接受数据的维度，也可以接受现成的数据
print("d:",d)
print("d:",d.type())

# 四维的tensor比较适合处理图片数据类型，cnn:[b,c,h,w]
e = torch.FloatTensor(2,2,3,4)  # 创建1片数据，每片数据2个通道，3行4列
print("e:",e)
print("e:",e.type())


f = torch.FloatTensor([2.,3.2])  # 大写的tensor接受数据的维度，也可以接受现成的数据
print("f:",f)
print("f:",f.type())
# 不推荐 torch.FloatTensor([2.,3.2]) = torch.tensor([2.,3.2])

g = torch.empty((2,3))  # 创建2*3的空数组，空数据中的值并不为0，而是未初始化的垃圾值，这些值非常随机。
print("g:",g)
print("g:",g.type())

# 未初始化的tensor一定要跟写入数据的后续步骤
# 增强学习一般使用double，其他一般使用float

torch.set_default_tensor_type(torch.DoubleTensor)
print("set_default_tensor_type:",torch.tensor([1.2,3]).type())
print(torch.tensor(1.))
print("type",torch.tensor(1.).type())

# 均匀采样0-10的TENSOR，要用x = 10*torch.rand(d1,d2)，randint只能采样整数额
# rand/rand_like,randint
h = 10 * torch.rand(3,3)  # 正态分布，N(0,1)，mean为0，std为1
print("torch.rand:",h)
print("torch.rand_like:",torch.rand_like(h))
print("torch.randint:",torch.randint(1,10,[3,3]))

# 自定义均值和方差
j = torch.normal(mean=torch.full([10],0),std=torch.arange(1,0,-0.1))
print("normal:",j)
print("生成一个值全为10的tensor:",torch.full([10],0))
print("生成一个等差数列:",torch.arange(1,0,-0.1))  # arange不包含右边界，左闭右开[1,0)
print("生成一个标量:",torch.full([], 2),",dim:",torch.full([], 2).dim(),",size:",torch.full([], 2).size())
print("生成一个向量:",torch.full([1], 2),",dim:",torch.full([1], 2).dim(),",size:",torch.full([1], 2).size())

# 等差数列
print(torch.arange(0,10))
print(torch.arange(0,10,2))
print(torch.range(0,10))  # 可以使用arange代替

# linspace/logspace
print(torch.linspace(0,10,steps=4))  # [0,10]
print(torch.linspace(0,10,steps=10))
print(torch.linspace(0,10,steps=11))

# logspace的base参数可以设置为2，10，e等底数
print(torch.logspace(0,-1,steps=10))  # 10^0,10^x,......10^-1

# ones/zeros/eye
print(torch.ones(3,3))  # 3x3矩阵元素全是1
print(torch.zeros(3,3))  # 3x3矩阵元素全是0
print(torch.eye(4,4))  # 4x4对角矩阵元素是1
print(torch.eye(4))
l = torch.zeros(3,3)
print("l:",torch.ones_like(l))

# randperm , random.shuffle
print("randperm:",torch.randperm(10)) # [0,9)
n = torch.rand(2,3)
m = torch.rand(2,2)
idx = torch.randperm(2)
idx
print("idx:",idx)
idx
print("idx:",idx)

print("n[idx]:",n[idx])
print("m[idx]:",m[idx])


test = torch.FloatTensor(1,2)
print(test)
col_types = {}
# col_types.append("'{0}':{1},".format(1,2))
col_types['1'] = 2
col_types['2'] = 3
col_types['3'] = 3
data = []
for k in col_types:
    data.append({k:col_types[k]})
print(data)