import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt

# Параметры задачи
L = 10           # Длина стержня
T = 2            # Время наблюдения
alpha = 0.01     # Коэффициент теплопроводности
Nx = 50          # Количество шагов по пространству
Nt = 1000        # Количество шагов по времени
dx = L / (Nx - 1)    # Шаг по пространству
dt = T / Nt      # Шаг по времени
g = 1            # Значение производной на правом конце

# Функция начального условия
def initial_condition(x):
    return np.sin(np.pi * x / L)

# Создаем сетку
x = np.linspace(0, L, Nx)
u = initial_condition(x)

# Матрица коэффициентов для неявной схемы
A = np.zeros((Nx, Nx))
np.fill_diagonal(A, 1 + 2*alpha*dt/dx**2)
np.fill_diagonal(A[:-1, 1:], -alpha*dt/dx**2)
np.fill_diagonal(A[1:, :-1], -alpha*dt/dx**2)

# Обрабатываем граничные условия
A[0, 0], A[0, 1] = 1, 0  # условие Дирихле слева
# Неймана справа, аппроксимация второго порядка, изменяем последнюю строку матрицы A
A[-1, -2], A[-1, -1] = -1, 1

# Решаем уравнение на каждом временном шаге
for n in range(Nt):
    # Правая часть системы
    b = u
    b[-1] = g * dx  # учет условия Неймана
    u = solve(A, b)

# Визуализация результата
plt.plot(x, u, label='Температура в момент времени T')
plt.xlabel('x')
plt.ylabel('u(x, T)')
plt.title('Распределение температуры в стержне')
plt.legend()
plt.show()
