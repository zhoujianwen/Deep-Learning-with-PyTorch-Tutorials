#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/18 17:20
# @Author  : zhoujianwen
# @Email   : zhou_jianwen@qq.com
# @File    : MergeSplit.py
# @Describe: 


# Merge or split
## Cat
### Statistics about scores
### [class1-4,students,scores]
### [class5-9,students,scores]
import torch
a = torch.rand(4,32,8)
b = torch.rand(5,32,8)
print("torch.cat([a,b],dim=0).shape:",torch.cat([a,b],dim=0).shape)

a1 = torch.rand(4,3,32,32)
a2 = torch.rand(5,3,32,32)
print("torch.cat([a1,a2],dim=0).shape:",torch.cat([a1,a2],dim=0).shape)

# 报错:Sizes of tensors must match except in dimension 0,
# 是因为维度dim=1的shape不一样造成的，而cat维度的shape可以不一样。
a1 = torch.rand(4,3,32,32)
a2 = torch.rand(4,1,32,32)
# print(torch.cat([a1,a2],dim=0).shape)  报错
print("torch.cat([a1,a2],dim=1).shape:",torch.cat([a1,a2],dim=1).shape)

a1 = torch.rand(4,3,16,32)
a2 = torch.rand(4,3,16,32)
print("torch.cat([a1,a2],dim=2).shape:",torch.cat([a1,a2],dim=2).shape)

### Along distinct dim/axis
"""插入图片"""

## Stack
a1 = torch.rand(4,3,16,32)
a2 = torch.rand(4,3,16,32)
print(torch.stack([a1,a2],dim=2).shape)  # [4,3,2,16,32]

a = torch.rand(32,8)
b = torch.rand(32,8)
print(torch.stack([a,b],dim=0).shape) # [2,32,8]
print(torch.cat([a,b],dim=0).shape) # [64,8]
## Stack与Cat最根本区别在于是联合还是合并。举一个简单例子，把a看作班级32个同学8门课程的成绩，
## 把b看作另一个班级32个同学8门课程的成绩，用Stack看作是两个班级的联合[2,32,8]，而用cat看作两个班级是一个整体[64,8]。
## 对于Stack而言两个维度都必须一致，而对于Cat而言拼接的那个维度可以不一样
# b = torch.rand([30,8])
## print(torch.stack([a,b],dim=0)) 报错
# print(torch.cat([a,b],dim=0).shape)

## Split
a = torch.rand(32,8)
b = torch.rand(32,8)
c = torch.stack([a,b],dim=0)
print(c.shape)

## 按长度拆分
## 长度不一样，可以直接给定一个list,[1,1]切片，其实就是拆成2块，每块长度是1。
## 如果是给定[1,2,3]就代表拆成3块，每块长度分别是1，2，3
aa,bb = c.split([1,1],dim=0)
print(aa.shape,bb.shape)
## 长度一样就设一个固定值，每块长度是1，拆分成n块，
aa,bb = c.split(1,dim=0)
print(aa.shape,bb.shape)

# ValueError: not enough values to unpack (expected 2, got 1)
# 拆分成n块，每块长度是2，但是c只能拆成1个，所以返回1个tensor，不能用2个tensor接受
# aa,bb = c.split(2,dim=0)

## Chunk
## 按数量拆分，拆分成2块，每块长度是2/2
aa,bb = c.chunk(2,dim=0)
print(aa.shape,bb.shape)

