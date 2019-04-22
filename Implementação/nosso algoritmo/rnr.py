import tensorflow as tf # para Deep Learning
import pandas as pd # para manipulação de dados
import numpy as np # para manipulação de matrizes
from matplotlib import pyplot as plt # para gráficos
from tensorflow.contrib.rnn import BasicRNNCell # para RNRs

bike_data = pd.read_csv('D:/RNR/hour1.csv') # lê os dados
bike_data.sort_values(["dteday", "hr"], inplace=True) # ordena temporalmente

demanda = bike_data[['cnt']] # pega a coluna de demanda
n_steps = 12 # define a quantidade de períodos de tempo

# cria n_steps colunas com a demanda defasada.
for time_step in range(1, n_steps+1):
    demanda['cnt'+str(time_step)] = demanda[['cnt']].shift(-time_step).values

demanda.dropna(inplace=True) # deleta linhas com valores nulos

X = demanda.iloc[:, :n_steps].values
X = np.reshape(X, (X.shape[0], n_steps, 1)) # adiciona dimensão

y = demanda.iloc[:, 1:].values
y = np.reshape(y, (y.shape[0], n_steps, 1))

print("parte 1",X.shape, y.shape)

n_test = 107

# obs: indexação negativa no Python é indexação de trás para frente
X_train, X_test = X[:-n_test, :, :], X[-n_test:, :, :]
y_train, y_test = y[:-n_test, :, :], y[-n_test:, :, :]

shuffle_mask = np.arange(0, X_train.shape[0]) # cria array de 0 a n_train
np.random.shuffle(shuffle_mask) # embaralha o array acima

# embaralha X e y consistentemente
X_train = X_train[shuffle_mask]
y_train = y_train[shuffle_mask]


# ------ aqui foi --------


n_inputs = 1 # variáveis de entrada
n_neurons = 64 # neurônios da camada recursiva
n_outputs = 1 # variáveis de entrada
learning_rate = 0.001 # taxa de aprendizado

graph = tf.Graph()
with graph.as_default():

    # placeholders
    tf_X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='X')
    tf_y = tf.placeholder(tf.float32, [None, n_steps, n_outputs], name='y')

    with tf.name_scope('Recurent_Layer'):
        cell = BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)
        outputs, last_state = tf.nn.dynamic_rnn(cell, tf_X, dtype=tf.float32)

    with tf.name_scope('out_layer'):
        stacked_outputs = tf.reshape(outputs, [-1, n_neurons])
        stacked_outputs = tf.layers.dense(stacked_outputs, n_outputs, activation=None)
        net_outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

    with tf.name_scope('train'):
        loss = tf.reduce_mean(tf.abs(net_outputs - tf_y)) # MAE
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

n_iterations = 12000
batch_size = 64

with tf.Session(graph=graph) as sess:
    init.run()

    for step in range(n_iterations+1):
        # cria os mini-lotes
        offset = (step * batch_size) % (y_train.shape[0] - batch_size)
        X_batch = X_train[offset:(offset + batch_size), :]
        y_batch = y_train[offset:(offset + batch_size)]

        # roda uma iteração de treino
        sess.run(optimizer, feed_dict={tf_X: X_batch, tf_y: y_batch})

        # mostra o MAE de treito a cada 2000 iterações
        if step % 2000 == 0:
            train_mae = loss.eval(feed_dict={tf_X: X_train, tf_y: y_train})
            print(step, "\tTrain MAE:", train_mae)

    # mostra o MAE de teste no final do treinamento
    test_mae = loss.eval(feed_dict={tf_X: X_test, tf_y: y_test})
    print(step, "\tTest MAE:", test_mae)

    # realiza previsões
    y_pred = sess.run(net_outputs, feed_dict={tf_X: X_test})
	
# --- inicio plota grafico -----	
	
sample = 10
n = 100 

plt.style.use("ggplot")
f = plt.figure(figsize=(10,7))
plt.plot(range(n), y_pred[:n,sample,0], label="Real")
plt.plot(range(n), y_test[:n,sample,0], color="yellow", label="Prevista")
plt.ylabel("Demanda (qtd)")
plt.xlabel("Tempo")
plt.legend(loc="best")
f.savefig("rnn_demanda1.png")
plt.show()	
	
from sklearn.metrics import r2_score
r2_score(y_pred=y_pred.reshape(-1,1), y_true=y_test.reshape(-1,1))

# --- fim plota grafico -----	


# ------------ Código que seria utilizado com uma segunda variável ------------


# features = ['cnt','temp','hum'] # variáveis preditivas
# demanda = bike_data[features]
# n_steps = 20

# for var_col in features: # para cada variável
    # for time_step in range(1, n_steps+1): # para cada período
        # # cria colunas da variável defasada
        # demanda[var_col+str(time_step)] = demanda[[var_col]].shift(-time_step).values

# demanda.dropna(inplace=True)

# n_var = len(features)
# columns = list(filter(lambda col: not(col.endswith("%d" % n_steps)), demanda.columns))

# X = demanda[columns].iloc[:, :(n_steps*n_var)].values
# X = np.reshape(X, (X.shape[0], n_steps, n_var))
# print(X.shape, y.shape)