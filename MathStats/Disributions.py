import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, chi2, f, norm

# Распределение хи квадрат с k степенями свободы

k = 5                                                # число степеней свободы (degrees of freedom)
chi2.cdf(0.7, df = k)                                # F(0.7) = P(X <= 0.7)    - функция распределения в точке 0.7
chi2.pdf(0.7, df = k)                                # f(0.7) = F'(0.7)        - функция плотности в точке 0.7
chi2.mean(df = k)                                    # E(X)                    - математическое ожидание X
chi2.var(df = k)                                     # Var(X)                  - дисперсия X
chi2.median(df = k)                                  # Median(X)               - медиана X
chi2.ppf(q = 0.95, df = k)                            # q: P(X < q) = 0.6       - квантиль уровня 0.6 с.в. X
#chi2.rvs(size = 1000, df = 5)                         #                         - выборка объема 1000 из X
print(chi2.mean(df = k),chi2.var(df = k))
print(chi2.median(df = k) ,chi2.ppf(q = 0.95, df = k))

# Графики

k=10
x = np.linspace(0, chi2.ppf(q = 0.999, df = k), 100)
#x= np.arange(0, chi2.ppf(q = 0.99, df = k), 1.0)
f_x = chi2.pdf(x, df = k)                            # значение функции плотности в соответствующих точках
F_x = chi2.cdf(x, df = k)                            # значение функции распределения в соответствующих точках
plt.xlabel('x')                                       # название нижней оси графика
plt.ylabel('f(x),F(x)')                                 # название верхней оси графика
plt.plot(x,f_x)                                         # график функции плотности
plt.plot(x,F_x)                                          # график функции распределения
plt.show();
chi2.ppf(q = 0.5, df = k)

# Распределение стьюдента

k = 5                                                # число степеней свободы
t.cdf(0.7, df = k)                                   # F(0.7) = P(X <= 0.7)    - функция распределения в точке 0.7
t.pdf(0.7, df = k)                                   # f(0.7) = F'(0.7)        - функция плотности в точке 0.7
t.mean(df = k)                                       # E(X)                    - математическое ожидание X
t.var(df = k)                                        # Var(X)                  - дисперсия X
t.median(df = k)                                     # Median(X)               - медиана X
#t.moment(n = 5, df = k)                              # E(X ^ 5)                - пятый (не центральный) момент X
t.ppf(q = 0.6, df = k)                               # q: P(X < q) = 0.6       - квантиль уровня 0.6 с.в. X
#t.rvs(size = 1000, df = 5)                            #                         - выборка объема 1000 из X
alpha = 2                                           # рассмотрим произвольную точку
# убедимся, что функции распределения обоих распредедений в этой точке очень близки
print(t.cdf(alpha, df = 10000) ,norm.cdf(alpha) )

# Графики

k=8
x = np.linspace(t.ppf(q = 0.001, df = k),            # точки, между которыми будет
                t.ppf(q = 0.999, df = k),            # строиться график
                100)                                 # количество точек (чем больше, тем больше детализация)
f_x = t.pdf(x, df = k)                               # значение функции плотности в соответствующих точках
F_x = t.cdf(x, df = k)                               # значение функции распределения в соответствующих точках
plt.xlabel('x')                                       # название нижней оси графика
plt.ylabel('f(x),F(x)')                                    # название верхней оси графика
plt.plot(x,f_x)                                         # график функции плотности
plt.plot(x,F_x)                                          # график функции распределения
plt.show();

# Распределение Фишера

k1 = 5                                               # число степеней свободы
k2 = 10
f.cdf(0.7, dfn = k1, dfd = k2)                      # F(0.7) = P(X <= 0.7)    - функция распределения в точке 0.7
f.pdf(0.7, dfn = k1, dfd = k2)                      # f(0.7) = F'(0.7)        - функция плотности в точке 0.7
f.mean(dfn = k1, dfd = k2)                          # E(X)                    - математическое ожидание X
f.var(dfn = k1, dfd = k2)                           # Var(X)                  - дисперсия X
f.median(dfn = k1, dfd = k2)                        # Median(X)               - медиана X
#f.moment(n = 3, dfn = k1, dfd = k2)                 # E(X ^ 3)                - третий (не центральный) момент X
f.ppf(q = 0.6, dfn = k1, dfd = k2)                  # q: P(X < q) = 0.6       - квантиль уровня 0.6 с.в. X
#f.rvs(size = 1000, dfn = k1, dfd = k2)              #                         - выборка объема 1000 из X

# Рассмотрим квантиль соответствующего уровня
alpha = 0.7
# Произведение соответствующих квантилей всегда
# будет равняться единице (догадайтесь почему)
f.ppf(q= 1 - alpha, dfn = k1, dfd = k2) * f.ppf(q = alpha, dfn = k2, dfd = k1)

# Графики

k1=5
k2=10
x = np.linspace(f.ppf(q = 0.001,
                      dfn = k1, dfd = k2),          # точки, между которыми будет
                f.ppf(q = 0.999,                      # строиться график
                      dfn = k1, dfd = k2),
                100)                                 # количество точек (чем больше, тем больше детализация)
f_x = f.pdf(x, dfn = k1, dfd = k2)                  # значение функции плотности в соответствующих точках
F_x = f.cdf(x, dfn = k1, dfd = k2)                  # значение функции распределения в соответствующих точках
plt.xlabel('x')                                       # название нижней оси графика
plt.ylabel('f(x),F(x)')                                    # название верхней оси графика
plt.plot(x,f_x)                                         # график функции плотности
plt.plot(x,F_x)                                          # график функции распределения
plt.show();
