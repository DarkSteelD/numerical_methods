from decimal import Decimal, getcontext
import numpy as np

# Установка точности вычислений до 50 знаков после запятой
getcontext().prec = 150

def decimal_atan(x, iterations=30):
    """Приближенное вычисление арктангенса для Decimal с использованием ряда Тейлора."""
    atan = Decimal(0)
    power_x = x
    for n in range(iterations):
        term = power_x / (2 * n + 1)
        if n % 2:
            atan -= term
        else:
            atan += term
        power_x *= x ** 2
    return atan
def decimal_matrix_multiply(A, B):
    n = len(A)
    result = np.zeros((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            sum = Decimal(0)
            for k in range(n):
                product =  A[i][k] * B[k][j]
                if product > Decimal('1E+50'):  # Пример условия
                    product = product.scaleb(-50)  # Масштабирование
                sum += product
            result[i][j] = sum
    return result
def decimal_cos(x, iterations=30):
    """Приближенное вычисление косинуса для Decimal."""
    cos_x = Decimal(1)
    term = Decimal(1)
    n = 1
    while n < iterations:
        term *= x ** 2 / (2 * n * (2 * n - 1))
        if n % 2:
            cos_x -= term
        else:
            cos_x += term
        n += 1
    return cos_x

def decimal_sin(x, iterations=30):
    """Приближенное вычисление синуса для Decimal."""
    sin_x = x
    term = x
    n = 1
    while n < iterations:
        term *= x ** 2 / (2 * n * (2 * n + 1))
        if n % 2:
            sin_x -= term
        else:
            sin_x += term
        n += 1
    return sin_x

def jacobi(A, E=Decimal('1e-10'), max_iterations=10000):
    n = A.shape[0]
    V = np.eye(n, dtype=object)  # Используем тип object для хранения Decimal
    for i in range(n):
        for j in range(n):
            V[i, j] = Decimal(V[i, j])
            A[i, j] = Decimal(str(A[i, j]))

    for _ in range(max_iterations):
        abs_A = np.abs(A - np.diag(np.diag(A)))
        i, j = np.unravel_index(np.argmax(abs_A), A.shape)
        if abs_A[i, j] < E:
            break

        if A[i, i] == A[j, j]:
            o = Decimal('0.25') * Decimal('3.14159265358979323846264338327950288419716939937510')
        else:
            o = Decimal('0.5') * decimal_atan(Decimal('2') * A[i, j] / (A[i, i] - A[j, j]))

        c = decimal_cos(o)
        s = decimal_sin(o)

        J = np.eye(n, dtype=object)
        for x in range(n):
            for y in range(n):
                J[x, y] = Decimal(J[x, y])
        J[i, i] = J[j, j] = c
        J[i, j] = s
        J[j, i] = -s

        A = decimal_matrix_multiply(J.T, A)
        A = decimal_matrix_multiply(A, J)

    eigenvalues = np.array([A[i, i] for i in range(n)])
    return eigenvalues, V

# Загрузка исходной матрицы A
A = np.array([
    [4, -30, 60, -35],
    [-30, 300, -675, 420],
    [60, -675, 1620, -1050],
    [-35, 420, -1050, 700]
], dtype=np.float64)

# Преобразование исходной матрицы A в Decimal
A_decimal = np.array([[Decimal(str(x)) for x in row] for row in A], dtype=object)

# Выполнение алгоритма
values, vectors = jacobi(A_decimal)
vectors_float = np.array(vectors, dtype=float)
print("Собственные значения:", values)
print("Собственные векторы:")

for i, vec in enumerate(vectors_float.T):
    print(f"Вектор {i + 1}: {vec}")
# Вывод собственных векторов не представлен, так как требуется дополнительная обработка для их корректного отображения