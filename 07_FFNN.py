from tensorflow.python.framework import graph_util
import tensorflow as tf
import numpy as np
import pickle
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def load_titles():
	return [fn[:-7] for fn in os.listdir('normalized_input_features/')]

def load_input_title(title):
	with open('normalized_input_features/' + title + '.pickle', "rb") as g: 
		return pickle.load(g)

def load_output_title(title):
	with open('normalized_output_features/' + title + '.bap_mgc', "rb") as g: 
		return pickle.load(g)

def weight(d1, d2):
	return tf.Variable(tf.random_normal([d1, d2],stddev=np.sqrt(1/d1)))

def combine(L, W):
	return tf.matmul(L, W)

def activate(x):
	alpha = 1.6732632423543772848170429916717
	scale = 1.0507009873554804934193349852946
	return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def calculate_rmse(prediction, target):
	return tf.sqrt(tf.reduce_mean(tf.squared_difference(prediction, target)))

def momentum_optimizer(loss):
	return tf.train.MomentumOptimizer(0.01, momentum=0.9, use_nesterov=True).minimize(loss)

def FFNN(D):

	X = tf.placeholder(tf.float32, [None, D[0]], name='X')
	W = [weight(D[i], D[i+1]) for i in range(len(D)-1)]

	L = []
	L_ = X
	for i in range(0, len(D)-2):
		_L = activate(combine(L_, W[i]))
		L.append(_L)
		L_ = _L

	Y_ = tf.identity(combine(L[-1], W[-1]), name='Y_')
	Y = tf.placeholder(tf.float32, [None, D[-1]])

	loss = calculate_rmse(Y_, Y)
	optimizer = momentum_optimizer(loss)

	return X, Y_, Y, loss, optimizer

titles = load_titles()
titles = titles[50:]

input_len = len(load_input_title(titles[0])[0])
output_len = len(load_output_title(titles[0])[0])

# Define dimensions of the neural network
X_d = input_len
Y_d = output_len

layer_size = 1024
n_layers = 16

h_layers = [layer_size for i in range(0, n_layers)]
D = [X_d]+h_layers+[Y_d]
print(D)

X, Y_, Y, loss, optimizer = FFNN(D)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	e = 0
	best_loss = math.inf

	while True:
		e += 1
		epoch_loss = 0
		f = open('FFNN_models/'+'output_log'+'.txt', 'a')

		for title in titles:

			input_vector = load_input_title(title)
			ouput_vector = load_output_title(title)

			_, current_loss = sess.run([optimizer, loss], {X:input_vector, Y:ouput_vector})
			epoch_loss += current_loss

		print('Epoch:', e)
		print('Loss:', epoch_loss)

		f.write('Epoch:' + str(e) + '\n')
		f.write('Loss:' + str(epoch_loss) + '\n\n')

		if epoch_loss < best_loss:
			best_loss = epoch_loss

			saver = tf.train.Saver()
			saver.save(sess, 'FFNN_models/'+'models')

			print('Saving frozen graph...')

			graph = tf.get_default_graph()
			input_graph_def = graph.as_graph_def()

			output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, ['Y_']) 

			with tf.gfile.GFile('FFNN_models/'+'frozen_model', "wb") as f:
				f.write(output_graph_def.SerializeToString())

		if epoch_loss == 0.0:
			break


