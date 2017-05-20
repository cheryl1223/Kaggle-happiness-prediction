import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

filename = '../data/train.csv'
df = pd.read_csv(filename, index_col = None, na_values=["?"])
df.replace('?', np.NaN)

data = df[['YOB', 'Income']].dropna()

mapping = {'under $25,000' : 12500, '$25,001 - $50,000' : 37500,
			'$50,000 - $74,999' : 62500, '$75,000 - $100,000' : 87500,
			'$100,001 - $150,000' : 125000, 'over $150,000' : 155000}

data = data.replace({'Income': mapping})

YOB = np.array(data['YOB'])
Income = np.array(data['Income'])

plt.scatter(YOB, Income, color='b', alpha=1, s=8)
plt.title("Scatter plot of YOB and income")
plt.xlabel("YOB (year of birth)")
plt.ylabel("The income")
plt.savefig("scatter_plot")
plt.show()
