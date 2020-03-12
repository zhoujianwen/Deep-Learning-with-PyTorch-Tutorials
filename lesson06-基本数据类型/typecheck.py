#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/11 13:57
# @Author  : zhoujianwen
# @Email   : zhou_jianwen@qq.com
# @File    : typecheck.py
# @Describe: 

import torch
import numpy as np
a = torch.randn(2,3)  # 随机正态分布生成二行三列的tensor数据，第一维是2，第二维是3，其dim是2
print("shape:",a.shape,"\nsize:",a.size())
print("type:",a.type())
print("第一个维度的信息，size:",a.size(0),"，shape:",a.shape[0])
print("第二个维度的信息:",a.size(1))
print("dim:",a.dim())

k = torch.rand(1,2,3)  # 使用随机均匀分布，数据分布范围在[0,1]
print("k:",k)
print("k[0]:",k[0])
print(list(k.shape))
print(torch.tensor(1.))
print(torch.tensor(1.3))
print(isinstance(a,torch.FloatTensor))
print(isinstance(a,torch.cuda.FloatTensor))
a = a.cuda()
print(isinstance(a,torch.cuda.FloatTensor))
b = torch.randn(2,3)
print(b)
print(len(b.shape))
print(b.shape)
print(b.size())
print("dim:",b.dim())
print(len(torch.tensor(1.).shape))
print(torch.tensor(1.).size())
print("------------------------------")
c = torch.FloatTensor(3)
print(c)
print(c.shape)  # size/shape表达tensor具体的形状
print(c.dim())  # dim表达size/shape的长度
print(len(c.shape))
print(torch.tensor([1.2,2,2]))
print(torch.tensor([1.2,2,2]).shape)
print(len(torch.tensor([1.2,2,2])))

print("torch.version:",torch.__version__)
data = np.ones(2)
print(data)
print(torch.from_numpy(data))

d = torch.rand(2,3,28,28)
print("size:",d.shape)
print("dim:",d.dim())
print("numel:",d.numel())  # numel是指tensor占用内存的数量 d.numel() = 2*3*28*28
print("numel = 2*3*28*28 = ",d.size(0)*d.size(1)*d.size(2)*d.size(3))
print("len:",len(d.shape))