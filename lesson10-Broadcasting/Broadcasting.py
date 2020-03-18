#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/18 14:09
# @Author  : zhoujianwen
# @Email   : zhou_jianwen@qq.com
# @File    : Broadcasting.py
# @Describe: 


# Broadcasting 自动扩展
## Expand
## without copying data

# Insert 1 dim ahead
# Expand dims with size 1 to same size

# 经过卷积神经网络之后得到一个Feature maps，有32个channel，在每一个channel上面添加一个Bias
# 如何让一个一维的Bias与四维的Feature maps叠加？Bias:[32] Expand 为 Bias:[32,1,1]
# [b,c,h,w] ，一般把后面的维度理解为小维度，前面的维度理解为大维度。比如b代表图片的数量，w代表
# 图片的列数，这样一对比就知道哪个概念小，哪个概念大。这样越靠前的维度就定义为大维度，越靠后的
# 维度就定义为小维度。对齐的时候是从小维度开始的。
# Feature maps:[4,32,14,14]
#         Bias:[  32, 1, 1]
# 根据Broadcasting意思会在dim=0插入1，使原来Bias变成四维Bias:[1,32,1,1]，最后Bias就能根据
# Feature maps维度扩展成Bias:[4,32,14,14]，其实就是在小维度1上面进行扩张。
# 总的来说基本过程是这样：Bias:[3,1,1] => [1,32,1,1] => [4,32,14,14]
#                               unsqueeze    expand
# 这样双方的dim、shape就都一样了。
# 若size一致，可进行对应位置元素相加

"""插入图片"""

# why broadcasting
## 1.for actual demanding
## [class,students,scores]
## Add bias for every students: +5 score
## [4,32,8] + [4,32,8]
## [4,32,8] + [5.0] ，此处的[5.0]是标量的意思，标量的dim可以理解为dim=0或dim=1也可以，要理解数据的
## 内容和数据的shape之间的区别，一个浮点数是4byte，本来[5.0]只占用1byte，如何人为将[5.0] 重复成 [4,32,8]，
## 那么这个tensor的大小就是4*32*8 = 1024byte，内存消耗比原来增加1000倍。若这个部分使用broadcasting就能很好节省内存，
## 这1000倍对于内存或显存的消耗来说就会产生很大的差距。
# 设计broadcasting初衷就是1满足自动扩张2满足数学要求3不用人为手动完成这项操作又节省内存空间。
# Is it broadcasting-able?
## Match from Last dim!
### if current dim=1,expand to same
### if either has no dim,insert one dim and expand to same
### otherwise,NOT broadcasting-able
### 小维度指定，大维度随意
### A:[4,32,8]
### B:[1, 1,4]
### A与B的这种情况就无法broadcasting

"""插入图片"""


## broadcasting-able
### Situation 1:
### [4,32,14,14]
### [1,32, 1, 1] => [4,32,14,14]
### Situation 2:
### [4,32,14,14]
### [14,14] => [1,1,14,14] => [4,32,14,14]
## NOT broadcasting-able
### Situation 1:
### A:[4,32,14,14]
### B:[2,32,14,14]
### 遇到这种情况只能手动操作，取出B[0] = [32,14,14] => [1,32,14,14] => [4,32,14,14]
### Dim 0 has dim,can NOT insert and expand to same
### Dim 0 has distinct dim,NOT size 1
### NOT broadcasting-able

import torch
import numpy as np
a = [[0,0,0],[10,10,10],[20,20,20],[30,30,30]]
a = torch.from_numpy(np.array(a))

b = torch.tensor([])

print(torch.range(0,2))

print(torch.tensor([5.0]).dim())