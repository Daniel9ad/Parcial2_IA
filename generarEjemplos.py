import pandas as pd
import numpy as np

data = pd.read_csv('winequality-red.csv', index_col=0)
print(data)
# Agrega ejemplos de entrenamiento--------------------------------
r3 = data.loc[data['quality']==3]
r4 = data.loc[data['quality']==4]
r5 = data.loc[data['quality']==5]
r6 = data.loc[data['quality']==6]
r7 = data.loc[data['quality']==7]
r8 = data.loc[data['quality']==8]

for i in range(len(r3)):
	n1 = r3.iloc[i:i+2,:]
	me = n1.mean()
	data = data.append(me, ignore_index=True)
print(data)
data.to_csv('winequality-red.csv')

# Cambia el orden las filas de manera aleatoria-----------------------
#df_random = data.sample(frac=1).reset_index(drop=True)
#df_random.to_csv('winequality-red.csv')

#col = data.index.tolist()
#random.shuffle(col)
#data1 = data.reindex(col)
#data1.to_csv('winequality-red1.csv')