import numpy as np
import matplotlib.pyplot as plt

a, b = 0, 10
N = 10
h = (b - a) / N
a1, a2 = 1, -1
b1, b2 = 1, -15

p = lambda x: 1
f = lambda x: x**2 + x - 1
exact_solution = lambda x: x**2 - x

A_improved = np.zeros((N+1, N+1))
b_improved = np.zeros(N+1)

for i in range(1, N):
    x_i = a + i * h
    if i >= 2:
        A_improved[i, i-2] = -1 / (12*h**2)
    if i >= 1:
        A_improved[i, i-1] = 16 / (12*h**2)
    A_improved[i, i] = -30 / (12*h**2) - p(x_i)
    if i <= N - 2:
        A_improved[i, i+1] = 16 / (12*h**2)
    if i <= N - 3:
        A_improved[i, i+2] = -1 / (12*h**2)
    b_improved[i] = -f(x_i)

A_improved[0, 0] = -3 / (2*h) - a1
A_improved[0, 1] = 2 / h
A_improved[0, 2] = -1 / (2*h)
b_improved[0] = a2

A_improved[N, N-2] = 1 / (2*h)
A_improved[N, N-1] = -2 / h
A_improved[N, N] = 3 / (2*h) - b1
b_improved[N] = b2

y_improved = np.linalg.solve(A_improved, b_improved)

x = np.linspace(a, b, N+1)
y_exact = np.array([exact_solution(xi) for xi in x])

plt.figure(figsize=(12, 6))
plt.plot(x, y_improved, 'b-', label='Численное решение')
plt.plot(x, y_exact, 'r--', label='Точное решение')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Сравнение численного и точного решений')
plt.show()

error = np.abs(y_exact - y_improved)
max_error = np.max(error)
rmse = np.sqrt(np.mean(error**2))
print(f"Максимальная ошибка: {max_error}")
print(f"Среднеквадратичная ошибка (RMSE): {rmse}")
