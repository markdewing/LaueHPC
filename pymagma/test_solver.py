
import numpy as np
import solver


solver.init()
A_txt = '../python/kernel/A.0.txt'
b_txt = '../python/data/b.0.txt'
x_txt = '../python/solutions/x.0.txt'


Ain = np.loadtxt(A_txt)
print('A',Ain.shape)

# Switch to column-major order
#A = np.array(Ain,order='F')
A = np.copy(Ain,order='F')
A_copy = np.copy(A)

b = np.loadtxt(b_txt)
print('b',b.shape)
b_copy = np.copy(b)

# Default method is qr
#x = solver.solve(A, b, place="gpu_simple")
x = solver.solve(A, b, place="cpu", method="svd")

print('x shape = ',x.shape)
#print('x = ',x[0:4])

soln = np.loadtxt(x_txt)
diff = np.sum((x-soln)**2)
print('diff =',diff)

residual = np.sum((np.dot(A_copy, x) - b_copy)**2)
print('residual = ',residual)

solver.fini()
