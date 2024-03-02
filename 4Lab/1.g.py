from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

def p(a):
    return 2
def f(a):
    return 3*np.sin(a)
def diff_eq(x, y):
    return y[1], p(a)*y[0] - f(x)

def boundary_condition(ya, yb):
    return [ya[0] - A, yb[0] - B]

a = 0
b = np.pi /2
A = 0
B = 1
factor = 0.005
tolerance = 1e-10
initial_guess = 1


sol = solve_ivp(diff_eq, (a, b), [A, initial_guess], t_eval=np.linspace(a, b, 100))


while abs(sol.y[0, -1] - B) > tolerance:
    initial_guess -= factor * (sol.y[0, -1] - B)
    sol = solve_ivp(diff_eq, (a, b), [A, initial_guess], t_eval=np.linspace(a, b, 100))
    print('guess')
plt.figure(figsize=(12, 6))
plt.plot(sol.t, sol.y[0][::], 'y', label='Cтрельба')
plt.plot(sol.t, np.sin(sol.t), 'r--', label='Точное решение')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Сравнение аппроксимированных и точных решений системы ОДУ')
plt.legend()
plt.grid(True)
plt.show()
print(sol)