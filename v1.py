import numpy as np
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
def jacobi(A, E=1e-10, max_iterations=100000):
    n = A.shape[0]
    #print(A)
    V = np.eye(n)
    should_break = False
    for _ in range(max_iterations):
        if should_break:
            break
        i, j = np.unravel_index(np.argmax(np.abs(np.triu(A, 1))), A.shape)
        # i, j = np.unravel_index(np.argmax(np.abs(np.triu(A, 1))+np.abs(np.tril(A, -1))), A.shape)
        if not check_symmetric(A):
            print(A)
            print(_)
            should_break = True
            break 
        # for d in range(1,n):
        #     if should_break:
        #         break
        #     for ol in range(n-1,d,-1):
        #         if A[d,ol] != A[ol,d]:
        #             min_value = min(A[d, ol], A[ol, d])
        #             A[d, ol] = min_value
        #             A[ol, d] = min_value
        # for d in range(1,n):
        #     if should_break:
        #         break
        #     for ol in range(n-1,d,-1):
        #         if A[d,ol] != A[ol,d]:
        #             print(A[d,ol] ,A[ol,d])
        #             print(A)
        #             should_break = True
        #             break 
        
        #print((A[i, j]))
        if i == j or i >= j:
            break
        if np.abs(A[i, j]) < E:
            break
        if A[i, i] == A[j, j]:
            o = np.pi / 4
        else:
            o = 0.5 * np.arctan(2 * A[i, j] / (A[i, i] - A[j, j]))
        #print(np.triu(A, 1))
        c = np.cos(o)
        s = np.sin(o)
        J = np.eye(n)
        J[i, i] = c
        J[j, j] = c
        J[i, j] = s
        J[j, i] = -s
        A = J.T @ A @ J
        V = V @ J
        #print(A)
        if _ +1 == max_iterations:
            print('O')

    eigenvalues = np.diag(A)
    return eigenvalues, V


A = np.loadtxt('A.txt',dtype=np.float64)
if not np.allclose(A, A.T, atol=1e-10):
    print("Не симметричная матрица")
# print(np.abs(np.triu(A, 1))+np.abs(np.tril(A, -1)))
values, vectors = jacobi(A)

with open('ans.txt', 'w') as file:
    print(values,"- [lambda]\n", vectors, file=file)

for i in range(len(vectors)):
    col = vectors[:, i]
    print(A @ col - values[i] * col)