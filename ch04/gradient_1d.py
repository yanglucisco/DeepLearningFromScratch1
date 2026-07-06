# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x 


def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y
     
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

# 随便写点东西，测试git如何解决冲突

tf = tangent_line(function_1, 5)
y2 = tf(x)

# 测试git冲突解决方法11

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
input("按 Enter 键退出...")