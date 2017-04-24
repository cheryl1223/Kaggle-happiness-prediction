import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

filename = '../data/train.csv'
df = pd.read_csv(filename, index_col = None, na_values=["?"])
df.replace('?', np.NaN)

YOB = df['YOB'].dropna()
YOB = np.array(YOB)

types = np.unique(YOB)
axis = np.arange(np.min(types), np.max(types)+1)

plt.hist(YOB, bins=axis, normed=True)
plt.xlabel("YOB (year of birth)")
plt.ylabel("The probability density")
plt.title("Histogram of YOB")

plt.savefig('histogram.png')

plt.show()