
import scipy.optimize
import numpy as np

def batched_non_negative_least_squares(A, y, epsilon=1e-6):
    m, n, bs = A.shape
    x = np.zeros((n, bs))
    
    # Initialize w_b, P, and R
    w_b = np.zeros((n, bs))
    P = [set() for _ in range(bs)]
    R = [set(range(n)) for _ in range(bs)]

    for b in range(bs):
        A_b = A[:, :, b]
        y_b = y[:, b]
        x_b = x[:, b]

        w_b[:, b] = np.dot(A_b.T, (y_b - np.dot(A_b, x_b)))
        
    for b in range(bs):
        A_b = A[:, :, b]
        y_b = y[:, b]
        x_b = x[:, b]
        while R[b] and np.max(w_b[list(R[b]),b]) > epsilon:
            j = max(R[b], key=lambda j: w_b[j, b])
            P[b].add(j)
            R[b].remove(j)
            
            AP = A_b[:, list(P[b])]
            s_b = np.zeros(n)
            #sP = np.dot(np.linalg.inv(np.dot(AP.T, AP)), np.dot(AP.T, y_b))
            sP = np.linalg.solve(np.dot(AP.T, AP), np.dot(AP.T, y_b))
            s_b[list(P[b])] = sP
            
            while np.min(sP) <= 0:
                alpha = min((x_b[i] / (x_b[i] - s_b[i])) for i in P[b] if s_b[i] <= 0)
                x_b = x_b + alpha * (s_b - x_b)
                P[b] = {j for j in P[b] if x_b[j] > 0}
                AP = A_b[:, list(P[b])]
                #sP = np.dot(np.linalg.inv(np.dot(AP.T, AP)), np.dot(AP.T, y_b))
                sP = np.linalg.solve(np.dot(AP.T, AP), np.dot(AP.T, y_b))
                s_b = np.zeros(n)
                s_b[list(P[b])] = sP

            x[:, b] = s_b
            w_b[:, b] = np.dot(A_b.T, (y_b - np.dot(A_b, x_b)))
    
    return x



def test1():
    # Example usage:
    nb = 2
    A = np.zeros((3,2,nb))
    A0 = np.array([[1, 2], [3, 4], [5, 6]])
    A1 = np.array([[1, 3], [8, 4], [2, 6]])
    A[:,:,0] = A0
    A[:,:,1] = A1
    y0 = np.array([[7, 8, 9]])
    y1 = np.array([[2, 8, 9]])
    y = np.zeros((3,nb))
    y[:,0] = y0
    y[:,1] = y1
    #y = np.array([1, 2, 3])
    epsilon = 1e-6
    x = batched_non_negative_least_squares(A, y, epsilon)
    print(x)

    out = scipy.optimize.nnls(A[:,:,0],y[:,0])
    print('reference 0',out)
    out = scipy.optimize.nnls(A[:,:,1],y[:,1])
    print('reference 1',out)

def test2():
    # Batch size
    nb = 5
    m = 20
    n = 10
    A = np.zeros((m,n,nb))
    y = np.zeros((m,nb))
    xref = np.zeros((n,nb))

    for b in range(nb):
        At = np.random.randn(m, n)
        yt = np.random.randn(m)
        A[:,:,b] = At
        y[:,b] = yt
        x1,resid = scipy.optimize.nnls(At, yt)
        xref[:,b] = x1

    x2 = batched_non_negative_least_squares(A, y)
    #print('from scipy, x=\n',xref)
    #print('from chatgpt, x=\n',x2)

    # Test solution quality
    for b in range(nb):
        okay = np.allclose(xref[:,b], x2[:,b])
        print(b,okay)
        if not okay:
            print(xref[:,b])
            print(x2[:,b])


if __name__ ==  "__main__":
    #test1()
    test2()
