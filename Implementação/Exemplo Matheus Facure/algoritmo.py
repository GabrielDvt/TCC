#import tensorflow as tf # para Deep Learning
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
#from tensorflow.contrib.rnn import BasicRNNCell

n_steps = 20		#define a quantidade de períodos de tempo

bike_data = pd.read_csv('hour.csv')		#Faz a leitura dos dados
bike_data.sort_values(["dteday", "hr"], inplace=True)

demanda = bike_data[["cnt"]]

#cria n_steps colunas com a demanda defasada
for time_step in range(1, n_steps + 1):
	demanda['cnt'+str(time_step)] = demanda[['cnt']].shift(-time_step).values

#remove as linhas com valores nulo
demanda.dropna(inplace=True)

print(demanda)

x = demanda.iloc[:, :n_steps].values
x = np.reshape(x, (x.shape[0], n_steps, 1))

y = demanda.iloc[:, 1:].values
y = np.reshape(y, (y.shape[0], n_steps, 1))

print(x.shape, y.shape)

