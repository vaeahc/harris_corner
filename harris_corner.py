#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:52:26 2019

@author: vaeahc
"""
#harris corner detect

import cv2
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread('lena512color.tiff', 0)

#建立高斯加权窗
sigma = 1
width = 3 * sigma
width_total = 2 * width + 1
W = np.zeros((width_total, width_total))

for i in range(width_total):
    for j in range(width_total):
        W[i][j] = 1 / (2 * np.pi * sigma ** 2) * np.exp(-((i - width) ** 2 + (j - width) ** 2) / (2 * sigma ** 2))
W /= np.sum(W)

kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
I_x = cv2.filter2D(im, -1, kernel_x)
kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
I_y = cv2.filter2D(im, -1, kernel_y)

I_x2 = I_x ** 2
I_y2 = I_y ** 2
I_xy = I_x * I_y
#窗函数加权
I_x2 = cv2.filter2D(I_x2, -1, W)
I_y2 = cv2.filter2D(I_y2, -1, W)
I_xy = cv2.filter2D(I_xy, -1, W)

for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        
        #计算M矩阵
        M = np.array([[I_x2[i][j], I_xy[i][j]], [I_xy[i][j], I_y2[i][j]]])
        #特征表达式, k = 0.04
        f = np.linalg.det(M) - 0.04 * (M[0][0] + M[1][1]) ** 2