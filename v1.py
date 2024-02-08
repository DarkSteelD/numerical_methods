import numpy as np

def jacobi(A, E=1e-10, max_iterations=1000):
    n = A.shape[0]
    V = np.eye(n)  
    for _ in range(max_iterations):
        i, j = np.unravel_index(np.argmax(np.abs(np.triu(A, 1))), A.shape)
        if np.abs(A[i, j]) < E:
            break
        if A[i, i] == A[j, j]:
            o = np.pi / 4
        else:
            o = 0.5 * np.arctan(2 * A[i, j] / (A[i, i] - A[j, j]))

        c = np.cos(o)
        s = np.sin(o)
        J = np.eye(n)
        J[i, i] = c
        J[j, j] = c
        J[i, j] = s
        J[j, i] = -s
        A = J.T @ A @ J
        V = V @ J

    eigenvalues = np.diag(A)
    return eigenvalues, V

# Example symmetric matrix
A = np.array([[4, -2, 2],
              [-2, 4, 2],
              [2, 2, 4]])

eigenvalues, eigenvectors = jacobi(A)

print(eigenvalues," - Cобственные значения \n Матрица собственных векторов(матрица вращений) : \n ", eigenvectors)
