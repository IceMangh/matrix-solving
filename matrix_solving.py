import os
import time

mpl_config_dir = os.path.join(os.path.dirname(__file__), ".matplotlib")
os.makedirs(mpl_config_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def print_matrix(A, name="Matrix"):
    print(name + ":")
    for row in A:
        print(" ".join(f"{x:12.6f}" for x in row))
    print()


def print_vector(v, name="Vector"):
    print(name + ":")
    for x in v:
        print(f"{x:12.6f}")
    print()


def copy_matrix(A):
    return [row[:] for row in A]


def copy_vector(b):
    return b[:]


def mat_vec_mul(A, x):
    n = len(A)
    result = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i][j] * x[j]
        result[i] = s
    return result


def vector_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]


def norm_inf(v):
    return max(abs(x) for x in v)


def hilbert_matrix(n):
    A = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(1.0 / (i + j + 1))
        A.append(row)
    return A


def lu_decomposition(A):

    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0

    for i in range(n):
        # вычисляем элементы U[i][j]
        for j in range(i, n):
            s = 0.0
            for k in range(i):
                s += L[i][k] * U[k][j]
            U[i][j] = A[i][j] - s


        # вычисляем элементы L[j][i]
        for j in range(i + 1, n):
            s = 0.0
            for k in range(i):
                s += L[j][k] * U[k][i]
            L[j][i] = (A[j][i] - s) / U[i][i]

    return L, U


def solve_lu(L, U, b):
    n = len(L)

    # прямой ход: Ly = b
    y = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i][j] * y[j]
        y[i] = b[i] - s

    # обратный ход: Ux = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U[i][j] * x[j]
        x[i] = (y[i] - s) / U[i][i]

    return x


def gauss_classic(A, b):
    A = copy_matrix(A)
    b = copy_vector(b)
    n = len(A)

    # прямой ход
    for k in range(n):

        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

    # обратный ход
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += A[i][j] * x[j]
        x[i] = (b[i] - s) / A[i][i]

    return x


def gauss_pivot(A, b):
    A = copy_matrix(A)
    b = copy_vector(b)
    n = len(A)

    # прямой ход
    for k in range(n):
        # поиск строки с максимальным элементом
        max_row = k
        max_val = abs(A[k][k])
        for i in range(k + 1, n):
            if abs(A[i][k]) > max_val:
                max_val = abs(A[i][k])
                max_row = i

        # перестановка строк
        if max_row != k:
            A[k], A[max_row] = A[max_row], A[k]
            b[k], b[max_row] = b[max_row], b[k]

        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

    # обратный ход
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += A[i][j] * x[j]
        x[i] = (b[i] - s) / A[i][i]

    return x


def test_one_matrix(n):
    A = hilbert_matrix(n)

    x_true = [1.0] * n
    b = mat_vec_mul(A, x_true)

    # LU
    t1 = time.perf_counter()
    L, U = lu_decomposition(A)
    x_lu = solve_lu(L, U, b)
    t2 = time.perf_counter()
    lu_time = t2 - t1
    lu_error = norm_inf(vector_sub(x_lu, x_true))

    # Гаусс
    t1 = time.perf_counter()
    x_gauss = gauss_classic(A, b)
    t2 = time.perf_counter()
    gauss_time = t2 - t1
    gauss_error = norm_inf(vector_sub(x_gauss, x_true))

    # Гаусс с выбором главного элемента
    t1 = time.perf_counter()
    x_pivot = gauss_pivot(A, b)
    t2 = time.perf_counter()
    pivot_time = t2 - t1
    pivot_error = norm_inf(vector_sub(x_pivot, x_true))

    print(f"LU-разложение: время = {lu_time:.8f} сек, ошибка = {lu_error:.8e}")
    print(f"Гаусс обычный: время = {gauss_time:.8f} сек, ошибка = {gauss_error:.8e}")
    print(f"Гаусс с выбором элемента: время = {pivot_time:.8f} сек, ошибка = {pivot_error:.8e}")
    print()

    return {
        "n": n,
        "times": {
            "LU-разложение": lu_time,
            "Гаусс обычный": gauss_time,
            "Гаусс с выбором элемента": pivot_time,
        },
        "errors": {
            "LU-разложение": lu_error,
            "Гаусс обычный": gauss_error,
            "Гаусс с выбором элемента": pivot_error,
        },
    }


def plot_results(results):
    sizes = [item["n"] for item in results]
    method_names = list(results[0]["times"].keys())
    output_path = os.path.join(os.path.dirname(__file__), "matrix_solving_results.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for method_name in method_names:
        times = [item["times"][method_name] for item in results]
        errors = [max(item["errors"][method_name], 1e-18) for item in results]

        axes[0].plot(sizes, times, marker="o", linewidth=2, label=method_name)
        axes[1].plot(sizes, errors, marker="o", linewidth=2, label=method_name)

    axes[0].set_title("Время работы методов")
    axes[0].set_xlabel("Размер матрицы n")
    axes[0].set_ylabel("Время, сек")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    axes[1].set_title("Ошибка методов")
    axes[1].set_xlabel("Размер матрицы n")
    axes[1].set_ylabel("Норма ошибки")
    axes[1].set_yscale("log")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    fig.suptitle("Сравнение методов решения СЛАУ для матриц Гильберта", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"График сохранен в файл: {output_path}")



def main():
    results = []
    for n in [5, 13, 25, 40, 67, 102, 140, 170]:
        print(n)
        results.append(test_one_matrix(n))

    plot_results(results)


if __name__ == "__main__":
    main()
