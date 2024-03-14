import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, chi2, f, norm

k = 10
print ("РАСПРЕДЕЛЕНИЕ ХИ - КВАДРАТ С 10 СТЕПЕНЯМИ СВОБОДЫ -------------------")
print("P(X <= 1.5) = ", chi2.cdf(1.5, df = k))
print("P(X > 1.5) = ", 1 - chi2.cdf(1.5, df = k))
print("P(1.5 <= X <= 2) = ", chi2.cdf(2, df = k) - chi2.cdf(1.5, df = k))
print("E = ", chi2.mean(df = k))
print("Var = ", chi2.var(df = k))
print("Median = ", chi2.median(df = k))
print("Квантиль уровня 0.1 = ", chi2.ppf(q = 0.1, df = k))
print("------------------------------------------------")

print ("РАСПРЕДЕЛЕНИЕ СТЬЮДЕНТА С 10 СТЕПЕНЯМИ СВОБОДЫ -------------------")
print("P(X <= 1.5) = ", t.cdf(1.5, df = k))
print("P(X > 1.5) = ", 1 - t.cdf(1.5, df = k))
print("P(1.5 <= X <= 2) = ", t.cdf(2, df = k) - t.cdf(1.5, df = k))
print("E = ", t.mean(df = k))
print("Var = ", t.var(df = k))
print("Median = ", t.median(df = k))
print("Квантиль уровня 0.1 = ", t.ppf(q = 0.1, df = k))
print("------------------------------------------------")

print ("РАСПРЕДЕЛЕНИЕ ФИШЕРА С 10 И 8 СТЕПЕНЯМИ СВОБОДЫ -------------------")
k1 = 8
k2 = 10
print("P(X <= 1.5) = ", f.cdf(1.5, dfn = k1, dfd = k2))
print("P(X > 1.5) = ", 1 - f.cdf(1.5, dfn = k1, dfd = k2))
print("P(1.5 <= X <= 2) = ", f.cdf(2, dfn = k1, dfd = k2) - f.cdf(1.5, dfn = k1, dfd = k2))
print("E = ", f.mean(dfn = k1, dfd = k2))
print("Var = ", f.var(dfn = k1, dfd = k2))
print("Median = ", f.median(dfn = k1, dfd = k2))
print("Квантиль уровня 0.1 = ", f.ppf(q = 0.1, dfn = k1, dfd = k2))
print("------------------------------------------------")
