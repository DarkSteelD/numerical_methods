import numpy as np
import matplotlib.pyplot as plt


def f(t, y):
    return y


def adam_bashforth_4(f, t0, y0, h, n):
    t = np.zeros(n+1)
    y = np.zeros(n+1)

    t[0] = t0
    y[0] = y0

    for i in range(1, 4):
        t[i] = t[i-1] + h
        k1 = f(t[i-1], y[i-1])
        k2 = f(t[i-1] + h/2, y[i-1] + (h/2) * k1)
        k3 = f(t[i-1] + h/2, y[i-1] + (h/2) * k2)
        k4 = f(t[i-1] + h, y[i-1] + h * k3)
        y[i] = y[i-1] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    for i in range(4, n+1):
        t[i] = t[i-1] + h
        y[i] = y[i-1] + (h/24) * (55*f(t[i-1], y[i-1]) - 59*f(t[i-2], y[i-2]) + 37*f(t[i-3], y[i-3]) - 9*f(t[i-4], y[i-4]))

    return y


t0 = 0
y0 = 0  
h = 0.1  
n = 100  
t_end = t0 + h * n

t_exact = np.linspace(t0, t_end, n+1)
y_exact = np.sin(t_exact)

y_approx = adam_bashforth_4(f, t0, y0, h, n)

plt.plot(t_exact, y_exact, label='Точное значение')
plt.plot(t_exact, y_approx, label='Approximation')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()