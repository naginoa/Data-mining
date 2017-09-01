from datetime import datetime
import numpy as np
import sys


def pythonsum(n):
    a = [i for i in range(n)]
    b = [i for i in range(n)]
    c = []
    for i in range(len(a)):
        a[i] = i ** 2
        b[i] = i ** 3
        c.append(a[i]+b[i])
    return c


def numpysum(n):
    a = np.arange(n) ** 2
    b = np.arange(n) ** 3
    c = a + b
    return c


if __name__ == '__main__':
    size = 1000
    start = datetime.now()
    c = pythonsum(size)
    delay = datetime.now() - start
    print("python运算次幂结果后三个：",c[-3:])
    print("python运行时间：(毫秒)", delay.microseconds)

    start = datetime.now()
    c = numpysum(size)
    delay = datetime.now() - start
    print("numpy运算次幂结果后三个：",c[-3:])
    print("numpy运行时间：(毫秒)", delay.microseconds)