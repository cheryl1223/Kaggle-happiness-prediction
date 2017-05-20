import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

filename = '../data/train.csv'
df = pd.read_csv(filename, index_col = None, na_values=["?"])
df.replace('?', np.NaN)

level1 = df.loc[df['Income'] == 'under $25,000']
level2 = df.loc[df['Income'] == '$25,001 - $50,000']
level3 = df.loc[df['Income'] == '$50,000 - $74,999']
level4 = df.loc[df['Income'] == '$75,000 - $100,000']
level5 = df.loc[df['Income'] == '$100,001 - $150,000']
level6 = df.loc[df['Income'] == 'over $150,000']

N = 6
happiness = np.zeros((2,N))

happiness[0,0] = level1.loc[level1['Happy'] == 1].shape[0]
happiness[1,0] = level1.loc[level1['Happy'] == 0].shape[0]

happiness[0,1] = level2.loc[level2['Happy'] == 1].shape[0]
happiness[1,1] = level2.loc[level2['Happy'] == 0].shape[0]

happiness[0,2] = level3.loc[level3['Happy'] == 1].shape[0]
happiness[1,2] = level3.loc[level3['Happy'] == 0].shape[0]

happiness[0,3] = level4.loc[level4['Happy'] == 1].shape[0]
happiness[1,3] = level4.loc[level4['Happy'] == 0].shape[0]

happiness[0,4] = level5.loc[level5['Happy'] == 1].shape[0]
happiness[1,4] = level5.loc[level5['Happy'] == 0].shape[0]

happiness[0,5] = level6.loc[level6['Happy'] == 1].shape[0]
happiness[1,5] = level6.loc[level6['Happy'] == 0].shape[0]

sum_happiness = happiness.sum(0)

happiness[0,:] = happiness[0,:]/sum_happiness
happiness[1,:] = happiness[1,:]/sum_happiness
#print(happiness.sum(0))

ind = np.arange(N)
width = 0.25
fig, ax = plt.subplots()
rects1 = ax.bar(ind, happiness[0,:], width, color='gold')
rects2 = ax.bar(ind + width, happiness[1,:], width, color='lightskyblue')

ax.set_title("Bar chart of income and happiness")
ax.set_xlabel("Income levels")
ax.set_ylabel("The fraction of happy/unhappy people")
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('under $25,000', '$25,001-$50,000', '$50,000-$74,999',
	'$75,000-$100,000', '$100,001-$150,000','over $150,000'),rotation = 10, fontsize=8)
ax.legend((rects1[0], rects2[0]), ('happy', 'unhappy'), loc = 'upper left')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.2f' % (height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
plt.savefig("bar_chart")
plt.show()
