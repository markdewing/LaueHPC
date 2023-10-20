# Converted to Python by ChatGPT from https://en.wikipedia.org/wiki/Non-negative_least_squares
#  with some fixes


import numpy as np
import scipy.optimize


def non_negative_least_squares(A, y, epsilon=1e-6):
    m, n = A.shape
    x = np.zeros(n)  # Initialize the solution vector x with zeros
    P = set()  # Initialize the set P to store selected indices
    R = set(range(n))  # Initialize the set R with all indices
    w = np.dot(A.T, (y - np.dot(A, x)))
    wr = w[list(R)]

    while R and max(wr) > epsilon:
        max_dot_product = -float("inf")
        j_max = None

        for j in R:
            dot_product = w[j]
            if dot_product > max_dot_product:
                max_dot_product = dot_product
                j_max = j

        P.add(j_max)  # Add the selected index to P
        R.remove(j_max)  # Remove the selected index from R

        # Construct the submatrix AP consisting of columns corresponding to indices in P
        AP = A[:, list(P)]
        s = np.zeros(n)  # Initialize a vector s with zeros
        print("AP size", AP.shape)
        sP = np.linalg.inv(np.dot(AP.T, AP)).dot(
            np.dot(AP.T, y)
        )  # Compute the least squares solution for the selected indices

        s[list(P)] = sP
        while min(sP) <= 0:
            alpha = float("inf")  # Initialize alpha as positive infinity

            for i in P:
                if s[i] <= 0:
                    alpha_candidate = x[i] / (x[i] - s[i])
                    if alpha_candidate < alpha:
                        alpha = alpha_candidate

            # Update the solution vector x with the computed alpha
            x = x + alpha * (s - x)

            # Move indices from P to R if their corresponding elements in x become non-positive
            for i in P.copy():
                if x[i] <= 0.0:
                    P.remove(i)
                    R.add(i)

            # Recompute the submatrix AP and the least squares solution sP
            AP = A[:, list(P)]
            # print('   inner AP size',AP.shape)
            s[:] = 0.0
            sP = np.linalg.inv(np.dot(AP.T, AP)).dot(np.dot(AP.T, y))

            s[list(P)] = sP  # Update the subvector s with sP for the selected indices

        x = s  # Update the solution vector x with the non-negative least squares solution
        w = np.dot(A.T, (y - np.dot(A, x)))
        wr = w[list(R)]

    return x


def test1():
    # Example usage:
    A = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([7, 8, 9])
    epsilon = 1e-6
    x = non_negative_least_squares(A, y, epsilon)
    print(x)


def test2():
    A = np.random.randn(50, 40)
    y = np.random.randn(50)

    x1, resid = scipy.optimize.nnls(A, y)
    x2 = non_negative_least_squares(A, y)
    print("from scipy, x=", x1)
    print("from chatgpt, x=", x2)

    # Test solution quality
    print("Solution is correct:", np.allclose(x1, x2))


if __name__ == "__main__":
    test1()
    test2()
