
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

soln = np.loadtxt(x_txt)

perf = solver.PerfInfo()

test_single = True
if test_single:
    # Default method is qr
    #x = solver.solve(A, b, place="gpu_simple", perf=perf)
    x = solver.solve(A, b, place="cpu", method="svd", perf=perf)

    print("elapsed time (s) = ",perf.elapsed)
    #print('x shape = ',x.shape)
    #print('x = ',x[0:5])

    diff = np.sum((x-soln)**2)
    print('diff =',diff)

    residual = np.sum((np.dot(A_copy, x) - b_copy)**2)
    print('residual = ',residual)


test_batch = True
if test_batch:
    nbatch = 10

    A_shape = list(A.shape)
    A_shape.append(nbatch)
    print("batched shape for A = ",A_shape)
    A_batch = np.zeros(A_shape, order='F')

    b_shape = list(b.shape)
    b_shape.append(nbatch)
    b_batch = np.zeros(b_shape, order='F')

    # Make a copy in each item
    for ib in range(nbatch):
        A_batch[:,:,ib] = A_copy[:,:]
        b_batch[:,ib] = b_copy[:]

    xb = solver.solve_batch(A_batch, b_batch, place="cpu", method="svd", perf=perf)

    diff = 0.0
    for ib in range(nbatch):
        #print('ib x',ib,xb[0:5,ib])
        diff += np.sum((xb[:,ib]-soln)**2)
        #print('diff = ',diff)
    print('ave diff =',diff/nbatch)

    residual = 0.0
    for ib in range(nbatch):
        residual += np.sum((np.dot(A_copy, xb[:,ib]) - b_copy)**2)
    print('ave residual = ',residual/nbatch)

solver.fini()
