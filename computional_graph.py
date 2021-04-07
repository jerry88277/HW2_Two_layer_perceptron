# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:43:28 2021

@author: JerryDai
"""
import numpy as np
a = np.array([1,2,3]).reshape(1,3)
b = np.array([4,5,6]).reshape(3,1)
c = np.array(2)
d = np.array([[1],[1],[2]])


s1 = a.dot(b)
s2 = s1 + c
s3 = np.matmul(d, s2)
s3 = d.dot(s2)
s4 = np.exp(s3)
s5 = 1/s4


y_b = np.array([[1], [1], [1]])

# JB = np.eye(3)
# np.fill_diagonal(JB, -1/(s4**2))
# s4_b = np.matmul(JB, y_b)
# s4_b = y_b.dot((-1/(s4**2)).T)
s4_b = np.multiply((-1/(s4**2)), y_b)

# JB = np.eye(3)
# np.fill_diagonal(JB, np.exp(s3))
# s3_b = np.matmul(JB, s4_b)
s3_b = np.multiply(np.exp(s3), s4_b)

# JB = np.eye(3)
# np.fill_diagonal(JB, s2)
# s2_1_b = np.matmul(JB, s3_b)
s2_1_b = np.multiply(d, s3_b)

# JB = np.eye(3)
# np.fill_diagonal(JB, d)
# s2_2_b = np.matmul(JB, s3_b)
s2_2_b = np.multiply(s2, s3_b)

# JB = np.eye(3)
# np.fill_diagonal(JB, 1)
# s1_1_b = np.matmul(JB, s2_1_b)
s1_1_b = np.multiply(1, s2_1_b)

# JB = np.eye(3)
# np.fill_diagonal(JB, 1)
# s1_2_b = np.matmul(JB, s2_1_b)
s1_2_b = np.multiply(1, s2_1_b)

# JB = np.eye(3)
# np.fill_diagonal(JB, a)
# a_b = np.matmul(JB, s1_1_b)
a_b = np.multiply(b.T, s1_1_b.T)

# JB = np.eye(3)
# np.fill_diagonal(JB, b)
# b_b = np.matmul(JB, s1_1_b)
b_b = np.multiply(a.T, s1_1_b)
