import pandas as pd
import numpy as np
import random

data = pd.read_csv('Red2.csv', index_col=0)
print(data)

# Agrega ejemplos de entrenamiento--------------------------------

for i in range(len(data)-8000):
	n1 = data.iloc[i:i+2,:]
	me = n1.mean()
	me['Country'] = int(me['Country'])
	me['Region'] = int(me['Region']+random.randint(0, data['Region'].min()))
	me['Winery'] = int(me['Winery']+random.randint(0, data['Winery'].min()))
	me['Rating'] = me['Rating']+random.random()
	me['NumberOfRatings'] = int(me['NumberOfRatings']+random.randint(0,data['NumberOfRatings'].min()))
	data = data.append(me, ignore_index=True)

print(data)
#data.to_csv('Red2.csv')

# Cambia el orden las filas de manera aleatoria-----------------------
#df_random = data.sample(frac=1).reset_index(drop=True)
#df_random.to_csv('Red2.csv')

#col = data.index.tolist()
#random.shuffle(col)
#data1 = data.reindex(col)
#data1.to_csv('winequality-red1.csv')