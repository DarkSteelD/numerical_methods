import numpy as np

def jacobi(A, E=1e-12, max_iterations=100000):
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
        A = J @ A 
        V = V @ J

    eigenvalues = np.diag(A)
    return eigenvalues, V


A = np.loadtxt('A.txt')
print(A)
values, vectors = jacobi(A)

with open('ans.txt', 'w') as file:
    print(values,"- [lambda]\n", vectors, file=file)

for i in range(len(vectors)):
    col = vectors[:, i]
    print(A @ col - values[i] * col)