#coding = utf-8
import numpy as np


a = np.array([[1, 2, 3],[4, 5, 6]], np.int32)
print(a.shape)
#print(a.flags)
#print(a.data)
#print(a.base)
#print(a.item)
print(a.tolist())
print(a.dumps())

b = np.arange(12).reshape(4, 3)
print(b)
print(a.reshape(3, 2))

print(np.int32)

#index
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(x[1:7:2])

x = np.array([[[1],[2],[3]], [[4],[5],[6]]])
print(x.shape)
print(x)

x_part = np.array([[1], [2], [3]])
print(x_part.shape)
