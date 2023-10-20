
import numpy as np
import solver

A = np.array([[1, 2], [3, 4], [5, 6]],dtype=np.float32,order='F')
y = np.array([7, 8, 9],dtype=np.float32)

x = solver.solvef(A, y)
print(x)
