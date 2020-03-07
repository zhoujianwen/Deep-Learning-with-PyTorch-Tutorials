# -*- coding: utf-8 -*-
# @Time    : 2020/3/7 23:19
# @Author  : zhoujianwen
# @Email   : zhou_jianwen@qq.com
# @File    : gd.py
# @Describe: 如何利用numpy采用梯度下降的方法求解二元一次方程组

import numpy as np


# y = wx + b
# MSE均方误差(1/n)∑(f(x_i)-y_i)^2，预测值f(x_i)与真实值y_i之差的平方和的平均值，
# MSE可用来作为衡量预测的一个指标。
# 计算出loss值
def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))


# 梯度下降(gradient descent)
# loss = (w_current*x + b_current-y)^2
# (Δw,Δb) = (∂L/∂w,∂L/∂b)
# new_w = w_current - lr * Δw
# new_b = w_current - lr * Δb
def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((w_current * x) + b_current))  # 把所有点的梯度信息（偏导数的值）累加起来
        w_gradient += -(2 / N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)  # 更新参数（权值和偏置）
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]


# 反复迭代梯度信息获取最优参数值
def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        print("num_iterations:{0},current_w:{1},current_b:{2}".format(i, w, b))
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]


"""
梯度下降的batch、epoch、iteration的含义，现在用的优化器SGD是stochastic gradient descent的缩写，
但不代表一个样本就更新一回，还是基于mini-batch的。
batchsize：批大小。在深度学习中，一般采用SGD训练，即每次训练在训练集中取batchsize个样本训练；
iteration：1个iteration等于使用batchsize个样本训练一次；
epoch：1个epoch等于使用训练集中的全部样本训练一次，通俗的讲epoch的值就是整个数据集被轮几次。
比如训练集有500个样本，batchsize = 10 ，那么训练完整个样本集：iteration=50，epoch=1.
batch: 深度学习每一次参数的更新所需要损失函数并不是由一个数据获得的，而是由一组数据加权得到的，这一组数据的数量就是batchsize。
batchsize最大是样本总数N，此时就是Full batch learning；最小是1，即每次只训练一个样本，这就是在线学习（Online Learning）。当我们分批学习时，每次使用过全部训练数据完成一次Forword运算以及一次BP运算，成为完成了一次epoch。
"""
def run():
    # 一次性读取训练集的100个样本点，batchsize=1，iteration= 100，epoch = 1，也就定义了每个epoch读取100个样本点
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0  # initial y-intercept guess
    initial_m = 0  # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}"
          .format(initial_b, initial_m,
                  compute_error_for_line_given_points(initial_b, initial_m, points))
          )
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".
          format(num_iterations, b, m,
                 compute_error_for_line_given_points(b, m, points))
          )

if __name__ == '__main__':
    run()
