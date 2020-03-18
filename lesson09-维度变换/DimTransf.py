#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/14 22:46
# @Author  : zhoujianwen
# @Email   : zhou_jianwen@qq.com
# @File    : DimTransf.py
# @Describe: 


import torch
import numpy as np

# View/reshape
a = torch.randn(4,1,28,28)

print("a.shape:",a.shape)

# cnn : [b,c,h,w]
# 将shape为4,1,28*28 view成4,1*28*28，而view的前提是保证numel()一致即可,
# view操作是要满足物理意义的，否则就会造成数据污染或破坏。
# 现在这个物理意义的view操作是把通道和行、列合并在一起变成1*28*28，这样就把
# 所有数据都合在一起用一个784的向量来表示了，即[4,784]。
# 这样相当于忽略掉c和h、w的信息，都合并在一起来检验了，这种转换之后的数据比较适合
# 全连接层的输入
print("a.view(4,28*28):",a.view(4,28*28))
print("a.view(4,28*28).shape:",a.view(4,28*28).shape)

# 将b,c,h合并成一个维度信息[N,28]
print("a.view(4*28,28).shape:",a.view(4*28,28).shape)
# 将b,c合并成一个维度信息[N,28,28]
print("a.view(4*1,28,28).shape:",a.view(4*1,28,28).shape)
# b结果是通过a.view操作得到的，只知道b的tensor[4,784]，在不知道原来a的tensor信息状态下，强行通过b结果逆向还原a，只能还原成[b,h,w,c]=[4,28,28,1]，
# 原来a的存储维度信息[b,c,h,w]就被打乱成[b,h,w,c]，造成数据的污染和破坏，数据的存储/维度顺序非常重要，需要时刻记住。
# view操作比较容易造成维度信息丢失，使用时要多加注意。
b = a.view(4,784)  # Logic Bug
print(b.shape)
print("b.view(4,28,28,1):",b.view(4,28,28,1))

print("[4,1,28,28] = 4*1*28*28 = ",a.numel())
# 原来a.tensor([4,783])，现改为[4,783]，与原来a.tensor无法保持维度信息一致会报错的。
# print("a.view(4,784):",a.view(4,783))

# Squeeze/unsqueeze，[-a.dim()-1,a.dim()+1) = index
# 在0维度前插入1，则a.shape = [1,4,1,28,28]，此时多出的维度可以理解为集合1或组1有[4,1,28,28]，组1有四张图片，每张图片是1*28*28
print("a.shape:",a.shape)
print("a.unsqueeze(0).shape:",a.unsqueeze(0).shape)  #

# 这里的-1是指原来a最后一个维度，在-1维度后插入1，则a.shape = [4,1,28,28,1]
# 这样就增加另外一个概念，就比如说在原来维度后面增加一个方差或均值的属性，
# 其实并没有改变原来数据信息，只是改变了数据的理解方式
print("a.unsqueeze(-1).shape:",a.unsqueeze(-1).shape)

# [4,1,28,28] -> [4,1,1,28,28]
print("a.unsqueeze(-4).shape:",a.unsqueeze(-4).shape)

# [4,1,28,28] -> [1,4,1,28,28]
print("a.unsqueeze(-5).shape:",a.unsqueeze(-5).shape)

# [4,1,28,28] -> Dimension out of range
# print("a.unsqueeze(5).shape:",a.unsqueeze(5).shape)

# 下面tensor([1.2,2.3])整个维度是1，用list列表信息来初始化tensor，[1.2,2.3]是指数据本身，并不是指维度信息
a = torch.tensor([1.2,2.3])
print("np.array([1.2,2.3]).shape:",np.array([1.2,2.3]).shape)
print("a:",a)
print("a.dim:",a.dim())
print("a.numel = 1 行 x 2 列:",a.numel())


# 数据本身和数据理解方式不一样的问题
# tensor([[1.2000],
#        [2.3000]])
print("a.unsqueeze(-1):\n",a.unsqueeze(-1))
# torch.Size([2, 1])
print("a.unsqueeze(-1):\n",a.unsqueeze(-1).shape)
# tensor([[1.2000, 2.3000]])
print("a.unsqueeze(0):\n",a.unsqueeze(0))
# torch.Size([1, 2])
print("a.unsqueeze(0):\n",a.unsqueeze(0).shape)


# For example

# bias相当于给每个channel上的所有像素增加一个偏置
b = torch.rand(32)
f = torch.rand(4,32,14,14)
# print(b.unsqueeze(0).unsqueeze(2).unsqueeze(3).shape)
# print(b.unsqueeze(1).unsqueeze().shape)
b = b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
print("b:",b)

# 维度删减，挤压
# [1,32,1,1]
print("b.shape:\n",b.dim())
# [1,32,1,1] -> [32]
print("b.squeeze().shape:\n",b.squeeze().shape)

# [1,32,1,1] -> [32,1,1]
print("b.squeeze(0).shape:\n",b.squeeze(0).shape)
# [1,32,1,1] -> [1,32,1]
print("b.squeeze(-1).shape:\n",b.squeeze(-1).shape)
# [1,32,1,1] -> [1,32,1,1] , dim=1的维度的长度不是1，而是32，故不挤压。
print("b.squeeze(1).shape:\n",b.squeeze(1).shape)
# [1,32,1,1] -> [32,1,1]
print("b.squeeze(-4).shape:\n",b.squeeze(-4).shape)

# Expand/repeat
a = torch.rand(4,32,14,14)

# [1,32,1,1]
print("b.shape:",b.shape)
# [1,32,1,1] -> [4,32,14,14] ，dim索引0、2、3的长度都是1，故1->N是可以扩张✓，否则3->M是不能扩张会报错的✕
print("b.expand(4,32,14,14).shape",b.expand(4,32,14,14).shape)

# [1,32,1,1] -> [1,32,1,1] ， dim索引的长度是-1的，扩张之后与原来保持不变
print("b.expand(-1,32,-1,-1).shape",b.expand(-1,32,-1,-1).shape)

# [1,32,1,1] -> [1,32,1,1]，dim=3的长度是-4的，扩张之后也与原来保持不变
print("b.expand(-1,32,-1,-4).shape",b.expand(-1,32,-1,-4).shape)

# [1,32,1,1]
print("b.shape:",b.shape)

# [1,32,1,1] -> [4,32*32,1,1]
print("b.repeat(4,32,1,1).shape:",b.repeat(4,32,1,1).shape)

# [1,32,1,1] -> [4,32,1,1]
print("b.repeat(4,1,1,1).shape:",b.repeat(4,1,1,1).shape)

# [1,32,1,1] -> [4,32,32,32]
print("b.repeat(4,1,32,32).shape:",b.repeat(4,1,32,32).shape)

# Transpose/t/permute
a = torch.randn(3,4)
print("a矩阵:\n",a)

# t() expects a 2D tensor, but self is 3D or 4D
print("a矩阵的转置：\n",a.t())

# [4,3,32,32]
a = torch.rand(4,3,32,32)
print("a.shape:",a.shape)

# 报错，[4,3,32,32] -> [4,32,32,3] -> [4,3*32*32] -> [4,"3",32,32]，在经过第一个view操作之后将原来
# 维度信息[4,32,32,3]压缩为[4,3*32*32]，过程不可逆，无法将 [4,3*32*32] 还原成 [4,"3",32,32]，相当于"3"的维度信息丢失了，
# a1 = a.transpose(1,3).view(4,3*32*32).view(4,3,32,32)
# print("a2.transpose(1,3):",a1)

# 正确的方法
# [4,3,32,32] -> [4,32,32,3] -> [4,3*32*32] -> [4,32,32,3] -> [4,3,32,32]
# view会导致维度顺序关系变模糊，所以需要人为跟踪。
a2 = a.transpose(1,3).contiguous().view(4,3*32*32).view(4,32,32,3).transpose(1,3)
print("a2.transpose(1,3):",a2.shape)

print("a eq a2 :",torch.all(torch.eq(a,a2)))

# permute
# [b,h,w,c]是numpy存储图片的格式，需要这一步才能导出numpy
a = torch.rand(4,3,28,28)
print("a.shape:",a.shape)
# [4,3,28,28] -> [4,28,3,28]
print("a.transpose(1,3).shape:",a.transpose(1,3).shape)

# [4,3,28,32]
b = torch.rand(4,3,28,32)
print("b.shape:",b.shape)

# [4,3,28,32] -> [4,32,28,3]
print("b.transpose(1,3).shape:",b.transpose(1,3).shape)

# [4,3,28,32] -> [4,32,28,3] -> [4,28,32,3]
print("b.transpose(1,3).transpose(1,2).shape:",b.transpose(1,3).transpose(1,2).shape)

# [4,3,28,32] -> [4,28,32,3]
# [b,c,h,w] -> [b,h,w,c]
# [0,1,2,3] -> [0,2,3,1]
print("b.permute(0,2,3,1).shape:",b.permute(0,2,3,1).shape)