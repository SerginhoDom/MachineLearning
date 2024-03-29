import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, t, norm


# Функция для создания гистограммы выборки заданного объема
def plot_sample_histogram(ax, sample, distribution_name, sample_size):
    ax.hist(sample, bins=int(np.sqrt(sample_size)), density=True, color='skyblue', alpha=0.7)
    ax.set_title(f'{distribution_name} (n={sample_size})')
    ax.set_xlabel('Значение')
    ax.set_ylabel('Частота')


np.random.seed(0)
sample_size = 1000

# Хи-квадрат распределение с 10 степенями свободы
chi_square_sample = np.random.chisquare(df=10, size=sample_size)

# Распределение Стьюдента с 10 степенями свободы
t_sample = np.random.standard_t(df=10, size=sample_size)

# Нормальное распределение с мат. ожиданием 10 и СКО 5
normal_sample = np.random.normal(loc=10, scale=5, size=sample_size)

# Гистограмма
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(chi_square_sample, bins=30, density=True, color='skyblue', alpha=0.7)
plt.title('Хи-квадрат распределение (df=10)')
plt.xlabel('Значение')
plt.ylabel('Частота')

plt.subplot(1, 3, 2)
plt.hist(t_sample, bins=30, density=True, color='salmon', alpha=0.7)
plt.title('t-распределение (df=10)')
plt.xlabel('Значение')
plt.ylabel('Частота')

plt.subplot(1, 3, 3)
plt.hist(normal_sample, bins=30, density=True, color='lightgreen', alpha=0.7)
plt.title('Нормальное распределение (мат. ожидание=10, дисперсия=25)')
plt.xlabel('Значение')
plt.ylabel('Частота')

plt.tight_layout()
plt.show()

# Выборочная функция распределения (ЭФР)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(chi_square_sample, bins=30, density=True, cumulative=True, color='skyblue', alpha=0.7, histtype='step')
plt.title('ЭФР для Хи-квадрат распределения (df=10)')
plt.xlabel('Значение')
plt.ylabel('Вероятность')

plt.subplot(1, 3, 2)
plt.hist(t_sample, bins=30, density=True, cumulative=True, color='salmon', alpha=0.7, histtype='step')
plt.title('ЭФР для t-распределения (df=10)')
plt.xlabel('Значение')
plt.ylabel('Вероятность')

plt.subplot(1, 3, 3)
plt.hist(normal_sample, bins=30, density=True, cumulative=True, color='lightgreen', alpha=0.7, histtype='step')
plt.title('ЭФР для Нормального распределения (мат. ожидание=10, дисперсия=25)')
plt.xlabel('Значение')
plt.ylabel('Вероятность')

plt.tight_layout()
plt.show()

# Генерация выборок разного размера
sample_sizes = [10, 100, 1000]

fig, axs = plt.subplots(3, 3, figsize=(15, 15))

for i, sample_size in enumerate(sample_sizes):
    # Хи-квадрат распределение
    chi_square_sample = np.random.chisquare(df=10, size=sample_size)
    plot_sample_histogram(axs[0, i], chi_square_sample, 'Хи-квадрат', sample_size)

    # Распределение Стьюдента
    t_sample = np.random.standard_t(df=10, size=sample_size)
    plot_sample_histogram(axs[1, i], t_sample, 't', sample_size)

    # Нормальное распределение
    normal_sample = np.random.normal(loc=10, scale=5, size=sample_size)
    plot_sample_histogram(axs[2, i], normal_sample, 'Нормальное', sample_size)

plt.tight_layout()
plt.show()

# Параметры распределения хи-квадрат
df = 10
x = 5  # Точка, в которой мы вычисляем ЭФР

# Количество выборок
num_samples = 10000

# Генерация выборок из распределения хи-квадрат
samples = chi2.rvs(df, size=num_samples)

# Вычисление ЭФР для каждой выборки в точке x
ecdfs = np.array([np.mean(sample <= x) for sample in samples])

# Оценка математического ожидания и дисперсии выборочной функции распределения
mean_ecdf = np.mean(ecdfs)
var_ecdf = np.var(ecdfs)

# Теоретические значения мат. ожидания и дисперсии выборочной функции распределения
theoretical_mean = chi2.cdf(x, df)
theoretical_var = (theoretical_mean * (1 - theoretical_mean) / num_samples) * 10000

# Вывод результатов
print("Оценка математического ожидания выборочной функции распределения:", mean_ecdf)
print("Оценка дисперсии выборочной функции распределения:", var_ecdf)
print("Теоретическое значение мат. ожидания выборочной функции распределения:", theoretical_mean)
print("Теоретическое значение дисперсии выборочной функции распределения:", theoretical_var)
