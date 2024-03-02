import numpy as np
import matplotlib.pyplot as plt

def fx(x, y):
    dydx = np.zeros(2) 
    dydx[0] = 2*np.exp(2*x) + 3 
    dydx[1] = 6*np.exp(2*x) + 2  
    return dydx

x0, xn = -5, 5
y0 = np.array([np.exp(2*x0) + 3*x0 + 1, 3*np.exp(2*x0) + 2*x0 + 1])  
h = 0.01
n = int((xn - x0) / h)

x = np.linspace(x0, xn, n + 1)
y = np.zeros((n + 1, 2))
y[0, :] = y0

for i in range(3):
    k1 = fx(x[i], y[i, :])
    k2 = fx(x[i] + h/2, y[i, :] + h * k1 / 2)
    k3 = fx(x[i] + h/2, y[i, :] + h * k2 / 2)
    k4 = fx(x[i] + h, y[i, :] + h * k3)
    y[i + 1, :] = y[i, :] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

for i in range(3, n):
    y[i + 1, :] = y[i, :] + (h / 24) * (
        55 * fx(x[i], y[i, :]) - 
        59 * fx(x[i - 1], y[i - 1, :]) + 
        37 * fx(x[i - 2], y[i - 2, :]) - 
        9 * fx(x[i - 3], y[i - 3, :]))

plt.figure(figsize=(12, 6))

plt.plot(x, y[:, 0], 'b-', label='Аппроксимация y1')
plt.plot(x, y[:, 1], 'g-', label='Аппроксимация y2')

exact_y1_vals = np.exp(2*x) + 3*x + 1
exact_y2_vals = 3*np.exp(2*x) + 2*x + 1

plt.plot(x, exact_y1_vals, 'y--', label='Точное y1')
plt.plot(x, exact_y2_vals, 'r--', label='Точное y2')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Сравнение аппроксимированных и точных решений системы ОДУ')
plt.legend()
plt.grid(True)
plt.show()
