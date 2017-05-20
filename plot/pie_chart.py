import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

filename = '.././data/train.csv'
df = pd.read_csv(filename)

men = df.loc[df['Gender']=='Male']
#print(men.shape)
happy_men = men.loc[men['Happy'] == 1]
#print(happy_men.shape)

unhappy_men = men.loc[men['Happy'] == 0]

#print(unhappy_men.shape)
num_happy_men = happy_men.shape[0]
num_unhappy_men = unhappy_men.shape[0]
total_num_men = num_unhappy_men + num_happy_men



women = df.loc[df['Gender']=='Female']
#print(women.shape)
happy_women = women.loc[women['Happy'] == 1]
#print(happy_women.shape)

unhappy_women = women.loc[women['Happy'] == 0]

#print(unhappy_women.shape)
num_happy_women = happy_women.shape[0]
num_unhappy_women = unhappy_women.shape[0]
total_num_women = num_unhappy_women + num_happy_women
#print(num_happy_men)
plt.subplot(1,2,1)

# Data to plot
labels = 'happy', 'unhappy'
sizes = [num_happy_men, num_unhappy_men]
colors = ['gold','lightskyblue']
explode = (0.1, 0)  # explode 1st slice
plt.title("Male")
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')

plt.subplot(1,2,2)
labels = 'happy','unhappy'
sizes = [num_happy_women,num_unhappy_women]
colors = ['gold','lightskyblue']
explode = (0.1, 0)  # explode 1st slice
plt.title("Female")
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.savefig('pie_chart')
plt.show()
