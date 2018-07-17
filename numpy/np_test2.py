import numpy as np


a = np.array([[56.0, 0.0, 4.4],
             [1.2, 104.0, 52.0]])
#print a

cal = a.sum(axis=0)
print cal

percentage = 100*a/cal
print percentage

print np.array([[1,2],[3,4]])/np.array([1,2])