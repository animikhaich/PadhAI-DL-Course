import torch, os, time, gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# os.system('clear')
cuda = torch.device('cuda:0')

TOTAL_RANGE = 100000
"""
# Numpy time for calculation
start = time.time()
for i in range(TOTAL_RANGE):
    a = np.random.randn(10000, 10000)
    b = np.random.randn(10000, 10000)
    c = np.matmul(a,b)

print("Numpy Time:", time.time() - start)

# PyTorch CPU time for calculation
start = time.time()
for i in range(TOTAL_RANGE):z
    a = torch.randn(10000, 10000)
    b = torch.randn(10000, 10000)
    c = torch.matmul(a,b)

print("PyTorch CPU Time:", time.time() - start)
"""
# PyTorch GPU time for calculation

a = torch.randn(10000, 10000, device=cuda)
b = torch.randn(10000, 10000, device=cuda)
c = torch.matmul(a,b)

start = time.time()
for i in range(TOTAL_RANGE):
    c = torch.matmul(a,b)

print("PyTorch GPU Time:", time.time() - start)


# # Tensorflow GPU time for calculation
# a = tf.random.uniform([10000, 10000], minval=-5, maxval=5, dtype=tf.float32)
# b = tf.random.uniform([10000, 10000], minval=-5, maxval=5, dtype=tf.float32)
# c = tf.matmul(a,b)

# start = time.time()
# for i in range(TOTAL_RANGE):
#     c = tf.matmul(a,b)

# print("Tensorflow GPU Time:", time.time() - start)
