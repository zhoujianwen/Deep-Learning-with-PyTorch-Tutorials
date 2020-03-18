#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/13 14:27
# @Author  : zhoujianwen
# @Email   : zhou_jianwen@qq.com
# @File    : IndexSlice.py
# @Describe: 索引与切片

import torch
import numpy as np

# cnn:[b,c,h,w]
a = torch.rand(4,3,28,28)
print("a[1].shape",a[1].shape)

# 第0张图片第0个通道的size
print("a[0,0].shape",a[0,0].shape)

# 第0张图片第0个通道的第二行第四列的像素点，dim=0的标量数据
print("a[0,0,2,4]:",a[0,0,2,4])
print("a[0,0,2,4].size:",a[0,0,2,4].shape)
print("a[0,0,2,4].dim:",a[0,0,2,4].dim())

# select first/last N
print("a.shape:",a.shape)

# [0,2)只包含第0、1张图片，不包含第2张图片，左闭右开
print("a[:2].shape:",a[:2].shape)

# 读取第[0,2)张图片的第[0,1)个通道的所有像素点的数据
print("a[:2,:1,:,:].shape:",a[:2,:1,:,:].shape)

# 读取第[0,2)张图片的第[1,3)个通道开始，也就是取G,B通道
print("a[:2,1:,:,:].shape:",a[:2,1:,:,:].shape)

# 读取第[0,2)张图片的第[-1 -> 3)个通道开始，也就是取B通道
print("a[:2,-1:,:,:].shape:",a[:2,-1:,:,:].shape)

# [n:],n<-x,[:n],x->n
b = [1,2,3]
print(b[-1:])  # [3]
print(b[:2])  # [1,2]
print(b[2:])  # [3]

# select by steps,其实只有一种通用形式：start:end:step
# 隔行采样，0:28:等同于0:28:1
print("a[:,:,0:28:2,0:28:2].shape:",a[:,:,0:28:2,0:28:2].shape)
print("a[:,:,::2,::2].shape",a[:,:,::2,::2].shape)

# select by specific index
print("a.shape:",a.shape)
# 选择第0个维度的第[0,2]张图片
print("a.index_select:",a.index_select(0,torch.tensor([0,2])))

# 选择第1个维度的第[1,2]个通道的图片
print("a.index_select(1,torch.tensor([1,2])).shape:",a.index_select(1,torch.tensor([1,2])).shape)

# 选择第2个维度的第[0,28)行
print("a.index_select(2,torch.arange((28)).shape:",a.index_select(2,torch.arange(28)).shape)

# 选择第2个维度的第[0,8)行
print("a.index_select(2,torch.arange((8)).shape:",a.index_select(2,torch.arange(8)).shape)

# 选择所有维度信息, ...代表任意多的意思，等价于:,:,:,:  ，两个::代表隔行
print("a[...].shape:",a[...].shape)
print("a[:,:,:,:].shape:",a[:,:,:,:].shape)

# 当有...出现时，右边的索引需要理解为最右边，
# ...会根据shape的实际情况推测a的维度，比如a[0,...,::2]，中间的...就代表C和H，::2 隔列采样
print("a[0,...,::2].shape:",a[0,...,::2].shape)
print("a[0,:,:,::2].shape:",a[0,:,:,::2].shape)
print("a[0].shape:",a[0].shape)

# 选择第0张图片的所有维度信息
print("a[0,...].shape:",a[0,...].shape)

# 选择所有图片的第一个通道的维度信息
print("a[:,1,...]:",a[:,1,...].shape)
print("a[:,1]:",a[:,1].shape)

# 选择所有图片的第2列像素的维度信息
print("a[...,:2].shape:",a[...,:2].shape)

# select by mask , .masked_select()
x = torch.randn(3,4)
print("x:",x)
mask = x.ge(0.5)
print("mask:",mask)

# 大于0.5的元素个数是根据内容才能确定的，比如取出所有概率大于0.5的物体
print("torch.masked_select(x,mask):",torch.masked_select(x,mask))

# dim=1的不定长度，长度取决于有多少个元素大于0.5
print("torch.masked_select(x,mask).shape:",torch.masked_select(x,mask).shape)

# select by flatten index
# 先把[2,3],dim=2 -> [6],dim=1
src = torch.tensor([[4,3,5],[6,7,8]])
print(torch.take(src,torch.tensor([0,2,5])))
"""

# Pandas中把dataframe和np.array的相互转换
# dataframe转化成array , df=df.values
# array转化成dataframe , import pandas as pd  , df = pd.DataFrame(df)
import pandas as pd

c = np.random.random([2,9])  # 随机生成2行9列的矩阵数据
print(c)
df = pd.DataFrame(c)  # 将array转化成dataframe
print(df)
result = pd.concat([df[0],df[8]],axis=1)  # 选取某几列合并数据，计算std
print("result:\n",result)

# 关于numpy和pandas中std()函数的区别
print("result.std():\n",result.std(ddof=0))  # 要想正常计算pandas的std，需要建ddof设置为0即可；pandas的ddof默认为1；
print("计算全局标准差：\n",np.std(result))
print("result.values:\n",result.values)
print("计算每一列的标准差：\n",np.std(result.values,axis=0))
print("计算每一行的标准差：\n",np.std(result.values,axis=1))

print(np.sum(result,axis=0))

a = [([]*2) for i in range(2)]
print(a)

s = "默认 情况下， concat 方法 是 沿着 axis= 0 的 轴 向 生效 的， 生成 另一个 Series。 如果 你 传递 axis= 1， 返回 的 结果 则是 一个 DataFrame（ axis= 1 时 是 列）："
print(s.replace(' ',''))

df1 = pd.DataFrame({'lkey':['b','b','a','c','a','a','b'],'data1': range(7)})
df2 = pd. DataFrame({'rkey': ['a','b','d'],'data2': range(3)})
print(df1)
print(df2)
print(pd.merge(df1,df2,left_on='lkey',right_on='rkey'))


print(pd.merge(df1,df2,left_on='lkey',right_on='rkey',how='left'))

df1 = pd.DataFrame({'key':['b','b','a','c','a','a','b'],'data1': range(7)})
df2 = pd. DataFrame({'key': ['a','b','d'],'data2': range(3)})
print(df1)
print(df2)
print(pd.merge(df1,df2,on=['key'],how='outer'))
"""

