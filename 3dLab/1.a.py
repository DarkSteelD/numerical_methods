import numpy as np
import matplotlib.pyplot as plt
def f(x, y):
    return x**2 - 2*y
x0 = 0  
y0 = 1.33  
h = 0.01 
def exact_solution(x):
    return (x**3)/3 + (4/3)*np.exp(-2*x)
def euler_method(x0, y0, h, f):
    x_values = [x0]
    y_values = [y0]

    x = x0
    y = y0

    while x < 1: 
        y += h * f(x, y)
        x += h

        x_values.append(x)
        y_values.append(y)

    return x_values, y_values
x_values_h, y_values_h = euler_method(x0, y0, h, f)
x_values_h2, y_values_h2 = euler_method(x0, y0, h/2, f)
x_exact = np.linspace(x0, 1, 1000)
y_exact = exact_solution(x_exact)
error_h = np.abs(np.array(y_values_h) - exact_solution(np.array(x_values_h)))
error_h2 = np.abs(np.array(y_values_h2) - exact_solution(np.array(x_values_h2)))
print("Порядок точности метода Эйлера явного при h", error_h)
print("Порядок точности метода Эйлера явного при h", error_h2)
plt.figure(figsize=(10, 5))
plt.plot(x_exact, y_exact, label='Аналитическое решение')
plt.plot(x_values_h, y_values_h, 'o-', label='Численное решение (h = 0.01)')
plt.plot(x_values_h2, y_values_h2, 'o-', label='Численное решение (h = 0.005)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()