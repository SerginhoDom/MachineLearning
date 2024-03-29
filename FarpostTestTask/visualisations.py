import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/dkulagin/kartaslov/master/dataset/orfo_and_typos/letter.matrix.csv', sep=';', index_col='INDEX_LETTER')

sns.heatmap(df)

plt.show()