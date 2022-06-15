import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import skew

# Вариант 6


def main(n):

    """Равномерное распределение"""
    a, b = 5, 8
    x_uniform = np.random.uniform(a, b, n)

    print("Равномерное распределение: ")
    print("Среднее значение:", np.mean(x_uniform))
    print("Дисперсия:", np.var(x_uniform))
    print("Коэффициент ассиметрии:", skew(x_uniform))
    print("Коэффициент эксцесса :", kurtosis(x_uniform))

    count, bins, ignored = plt.hist(x_uniform, 10, edgecolor='k', density=True)
    plt.plot(bins, np.ones_like(bins) / (b - a), linewidth=2, color='r')
    plt.title('Равномерное распределение')
    plt.show()

    """Моделирование равномерного распределенной случайной величины c использованием линейного конгруэнтного метода"""
    def kon_method():
        m = 2 ** 31 - 1
        a = 16807
        c = 1
        seed = 12345
        x_u = np.zeros(n)

        x_u[0] = ((seed * a + 1) % m) / m
        for i in range(1, n):
            x_u[i] = ((a * x_u[i - 1] + c) % m)
            i += 1
    kon_method()

    """Нормальное распределение"""
    mu, sigma = 5, 6
    x_n = np.random.normal(mu, sigma, n)

    print("Нормальное распределение: ")
    print("Среднее значение:", np.mean(x_n))
    print("Дисперсия:", np.var(x_n))
    print("Коэффициент ассиметрии:", skew(x_n))
    print("Коэффициент эксцесса :", kurtosis(x_n))

    count, bins, ignored = plt.hist(x_n, 30, edgecolor='k', density=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=2, color='r')
    plt.title('Нормальное распределение')
    plt.show()

    """Моделирование нормального распределения с использованием преобразования Бокса-Мюллера"""
    uni1 = np.random.uniform(0, 1, n)
    uni2 = np.random.uniform(0, 1, n)
    z0 = (-2 * np.log(uni1)) ** 0.5 * np.cos(2 * np.pi * uni2)

    """Распределение Рэлея"""
    x_r = np.random.rayleigh(6, n)

    print("Распределение Рэлея: ")
    print("Среднее значение:", np.mean(x_r))
    print("Дисперсия:", np.var(x_r))
    print("Коэффициент ассиметрии:", skew(x_r))
    print("Коэффициент эксцесса :", kurtosis(x_r))

    count, bins, ignored = plt.hist(x_r, 14, edgecolor='k', density=True)
    plt.title('Распределение Рэлея')
    plt.show()

    """Моделирование распределения Рэлея"""
    x = np.random.normal(0, 1, n)
    y = np.random.normal(0, 1, n)
    reley = np.zeros(n)
    for i in range(0, n):
        reley[i] = ((x[i] ** 2 + y[i] ** 2) ** 0.5)
        i += 1

    """Распределение Пуассона"""
    x_p = np.random.poisson(6, n)

    print("Распределение Пуассона: ")
    print("Среднее значение:", np.mean(x_p))
    print("Дисперсия:", np.var(x_p))
    print("Коэффициент ассиметрии:", skew(x_p))
    print("Коэффициент эксцесса :", kurtosis(x_p))

    count, bins, ignored = plt.hist(x_p, 14, edgecolor='k', density=True)
    plt.title('Распределение Пуассона')
    plt.show()


print(f"Для N = 1000: {main(5000)}")
print(f"Для N = 10000: {main(50000)}")
print(f"Для N = 100000: {main(500000)}")


