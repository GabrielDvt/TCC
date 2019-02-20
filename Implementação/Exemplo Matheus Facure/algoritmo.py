import tensorflow as tf # para Deep Learning
import pandas as pd
import numpy as np
from tensorflow.contrib.rnn import BasicRNNCell # para RNRs
from matplotlib import pyplot as plt
#from tensorflow.contrib.rnn import BasicRNNCell

n_steps = 20		#define a quantidade de períodos de tempo
n_test = 500

#dados da rede
n_inputs = 1		#variáveis de entrada
n_neurons = 64	#neurônios na camada recursiva
n_outputs = 1	#variáveis de entrada
learning_rate = 0.001	#taxa de aprendizado

bike_data = pd.read_csv('hour.csv')		#Faz a leitura dos dados
bike_data.sort_values(["dteday", "hr"], inplace=True)

demanda = bike_data[["cnt"]]

#cria n_steps colunas com a demanda defasada
for time_step in range(1, n_steps + 1):
	demanda['cnt'+str(time_step)] = demanda[['cnt']].shift(-time_step).values

#remove as linhas com valores nulo
demanda.dropna(inplace=True)

#x = variáveis independentes
x = demanda.iloc[:, :n_steps].values

#adiciona 1 dimensão para x para ficar no formato [n_amostras, periodo, variáveis]
x = np.reshape(x, (x.shape[0], n_steps, 1))

#variáveis dependentes
y = demanda.iloc[:, 1:].values
y = np.reshape(y, (y.shape[0], n_steps, 1))

#prepara o array de teste 
x_train, x_test = x[:-n_test, :, :], x[-n_test:, :, :]
y_train, y_test = y[:-n_test, :, :], y[-n_test:, :, :]

#embaralha os dados de treino para utilizar o gradiente descendente estocástico
shuffle_mask = np.arange(0, x_train.shape[0])
np.random.shuffle(shuffle_mask)

x_train = x_train[shuffle_mask]
y_train = y_train[shuffle_mask]


graph = tf.Graph()
with graph.as_default():
	tf_x = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='x')
	tf_y = tf.placeholder(tf.float32, [None, n_steps, n_outputs], name='y')

	with tf.name_scope('Recurent_Layer'):
		cell = BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)
		outputs, last_state = tf.nn.dynamic_rnn(cell, tf_x, dtype=tf.float32)

	with tf.name_scope('out_layer'):
		stacked_outputs = tf.reshape(outputs, [-1, n_neurons])
		stacked_outputs = tf.layers.dense(stacked_outputs, n_outputs, activation=None)
		net_outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

	with tf.name_scope('train'):
		loss = tf.reduce_mean(tf.abs(net_outputs - tf_y))
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

	init = tf.global_variables_initializer()		

n_iterations = 10000
batch_size = 64

with tf.Session(graph=graph) as sess:
    init.run()
    
    for step in range(n_iterations+1):
        # cria os mini-lotes
        offset = (step * batch_size) % (y_train.shape[0] - batch_size)
        X_batch = x_train[offset:(offset + batch_size), :]
        y_batch = y_train[offset:(offset + batch_size)]
        
        # roda uma iteração de treino
        sess.run(optimizer, feed_dict={tf_x: X_batch, tf_y: y_batch})
    
        # mostra o MAE de treito a cada 2000 iterações
        if step % 2000 == 0:
            train_mae = loss.eval(feed_dict={tf_x: x_train, tf_y: y_train})
            print(step, "\tTrain MAE:", train_mae)
    
    # mostra o MAE de teste no final do treinamento
    test_mae = loss.eval(feed_dict={tf_x: x_test, tf_y: y_test})
    print(step, "\tTest MAE:", test_mae)

    # realiza previsões
    y_pred = sess.run(net_outputs, feed_dict={tf_x: x_test})

from sklearn.metrics import r2_score
r2_score(y_pred=y_pred.reshape(-1,1), y_true=y_test.reshape(-1,1))
