
import numpy as np
from scipy import optimize
from scipy import linalg
import time

# Different solution methods for Ax=b for an overdetermined system

A_txt = 'kernel/A.0.txt'
b_txt = 'data/b.0.txt'

A = np.loadtxt(A_txt)
print('A',A.shape)
b = np.loadtxt(b_txt)
print('b',b.shape)

#(x, res, rank, s) = np.linalg.lstsq(A,b,rcond=-1)
#print('out = ',x.shape,res[0])

use_nnls = True
if use_nnls:
    print("\nsolve using NNLS")
    start = time.perf_counter()
    out = optimize.nnls(A, b)
    end = time.perf_counter()

    print('solution time = ',end-start)
    print('solution shape = ',out[0].shape)
    residual = np.sum((np.dot(A, out[0]) - b)**2)
    print('NNLS residual = ',residual)
    #print('solution = ',out[0])
    #for x in out[0]:
    #    print(x)

use_LU = True
if use_LU:
    print("\nsolve using LU")
    lu,piv,info = linalg.lapack.dgetrf(A)
    print('lu shape = ',lu.shape)

    # Embed in a larger matrix/vector otherwise dgetrs will complain about dimensions

    #lu_superset = np.ones((798,798))
    lu_superset = np.eye(798)

    lu_superset[:,0:256] = lu[:,:]
    #b_superset = np.zeros((798))
    #b_superset[0:256] = b[:]
    piv_superset= np.zeros((798))
    piv_superset[0:256] = piv[:]

    print('lu shape = ',lu_superset.shape)
    x, info = linalg.lapack.dgetrs(lu_superset, piv_superset, b)
    print('x shape = ',x.shape)
    #print(x)

    diff = np.sum((x[0:256]-out[0])**2)
    print('LU diff =',diff)

    residual = np.sum((np.dot(A, x[0:256]) - b)**2)
    print('LU residual = ',residual)


use_QR = True
if use_QR:
    print("\nsolve using QR")
    start = time.perf_counter()
    qr,tau,work,info = linalg.lapack.dgeqrf(A)
    lwork = 1
    cq,work,info = linalg.lapack.dormqr("L","T",qr,tau,b,lwork)
    #print("cq",cq.shape)
    #print("cq",cq)

    r1 = np.triu(qr)
    #print('shape r1',r1.shape)

    r2 = r1[0:256,0:256]

    x = linalg.blas.dtrsm(1.0, r2, cq[0:256])
    #for xt in x:
    #    print(xt)
    #print(x)
    end = time.perf_counter()
    print("QR solution time= ",end-start)

    diff = np.sum((x - out[0])**2)
    print('QR diff = ',diff)

    residual = np.sum((np.dot(A, x) - b)**2)
    print('QR residual = ',residual)


use_SVD = True
if use_SVD:
    print("\nsolve using SVD")
    u,s,vt,info = linalg.lapack.dgesvd(A)
    print('u shape = ',u.shape, ' vt shape = ',vt.shape, ' s shape = ',s.shape)
    #print('singular values',s)
    inv_s = 1.0/s;
    Sinv = np.zeros((798,798))
    Sinv[0:256,0:256] = np.diag(inv_s)
    x = np.dot(vt.T, np.dot(Sinv, np.dot(u.T, b))[0:256])
    print('SVD solution shape',x.shape)
    print(' u.T,b shape',np.dot(u.T,b).shape)
    print(' inv_s . u.T,b shape',np.dot(Sinv, np.dot(u.T,b)).shape)

    diff = np.sum((x - out[0])**2)
    print('SVD diff = ',diff)

    residual = np.sum((np.dot(A, x) - b)**2)
    print('SVD residual = ',residual)
