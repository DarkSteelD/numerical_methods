import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.cos(t)

def adams_bashforth_4(f, t0, y0, h, n):
    y = np.zeros(n+1)
    t = np.linspace(t0, t0 + n*h, n+1)
    y[:4] = y0  
    for i in range(3, n):
        y[i+1] = y[i] + (h/24) * (55*f(t[i]) - 59*f(t[i-1]) + 37*f(t[i-2]) - 9*f(t[i-3]))
    
    return t, y


t0 = 0
h = 0.1  
y0 = [np.sin(t0), np.sin(t0 + h), np.sin(t0 + 2*h), np.sin(t0 + 3*h)] 
n = 100 

t_approx, y_approx = adams_bashforth_4(f, t0, y0, h, n)
t_exact = np.linspace(t0, t0 + n*h, 1000)
y_exact = np.sin(t_exact)

plt.figure(figsize=(10, 6))
plt.plot(t_exact, y_exact, label='Точное решение ($y = \sin(t)$)', color='blue', linestyle='--')
plt.plot(t_approx, y_approx, label='Приближенное решение неявным методом Адамса', color='red', marker='o', markersize=4, linestyle='none')
plt.title('Сравнение точного и приближенного решений')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
