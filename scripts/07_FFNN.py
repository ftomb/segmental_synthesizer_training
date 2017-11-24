from tensorflow.python.framework import graph_util
from random import shuffle
import tensorflow as tf
import numpy as np
import pickle
import math
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def load_titles(normalized_input_features_path, normalized_output_features_path):
	input_titles = [os.path.splitext(fn)[0] for fn in os.listdir(normalized_input_features_path) if os.path.splitext(fn)[1] == '.pickle']
	output_titles = [os.path.splitext(fn)[0] for fn in os.listdir(normalized_output_features_path) if os.path.splitext(fn)[1] == '.bap_mgc']
	return set(input_titles).intersection(output_titles)

def load_input_title(title, normalized_input_features_path):
	with open(os.path.join(normalized_input_features_path, title + '.pickle'), "rb") as g:
		return pickle.load(g)

def load_output_title(title, normalized_output_features_path):
	with open(os.path.join(normalized_output_features_path, title + '.bap_mgc'), "rb") as g:
		return pickle.load(g)

def weight(d1, d2):
	return tf.Variable(tf.random_normal([d1, d2],stddev=np.sqrt(1/d1)))

def rnn_layer(X, recurrent_nodes):
	rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=recurrent_nodes, activation=tf.nn.elu)
	return tf.nn.dynamic_rnn(cell=rnn_cell, inputs=X, dtype=tf.float32)

def combine(L, W):
	return tf.matmul(L, W)

def activate(L):
	alpha = 1.6732632423543772848170429916717
	scale = 1.0507009873554804934193349852946
	return scale*tf.where(L>=0.0, L, alpha*tf.nn.elu(L))

def calculate_rmse(prediction, target):
	return tf.sqrt(tf.reduce_mean(tf.squared_difference(prediction, target)))
	
def momentum_optimizer(loss):
	return tf.train.MomentumOptimizer(0.01, momentum=0.9, use_nesterov=True).minimize(loss)

def RNN(D):

	X = tf.placeholder(tf.float32, [None, D[0]], name='X')
	W = [weight(D[i], D[i+1]) for i in range(len(D)-1)]

	L = []
	L_ = X
	for i in range(0, len(D)-3):
		_L = activate(tf.matmul(L_, W[i]))
		L.append(_L)
		L_ = _L

	n_frames = tf.placeholder(tf.int32, name='n_frames')
	L1 = tf.reshape(L[-1], [-1, n_frames, D[-3]])

	outputs, states = rnn_layer(L1, D[-2])
	outputs = tf.reshape(outputs, [-1, D[-2]])

	Y_ = tf.matmul(outputs, W[-1], name='Y_')
	Y = tf.placeholder(tf.float32, [None, D[-1]], name='Y')

	loss = calculate_rmse(Y_, Y)
	optimizer = momentum_optimizer(loss)

	return X, n_frames, Y_, Y, loss, optimizer

if __name__ == '__main__':

	normalized_input_features_path = sys.argv[1]
	normalized_output_features_path = sys.argv[2]
	FFNN_models_path = sys.argv[3]

	titles = list(load_titles(normalized_input_features_path, normalized_output_features_path))
	print(titles)

	corpus_split = 1
	test_titles = sorted(titles)[:corpus_split]
	titles = sorted(titles)[corpus_split:]
	f = open(os.path.join(FFNN_models_path, 'test_titles'+'.txt'), 'a')
	f.write(str(test_titles))



	X_d = len(load_input_title(titles[0], normalized_input_features_path)[0])
	Y_d = len(load_output_title(titles[0], normalized_output_features_path)[0])


	# Define dimensions of the neural network

	forward_nodes = 500
	recurrent_nodes = 500
	forward_layers = 10
	h_layers = [forward_nodes for i in range(0, forward_layers)]
	D = [X_d]+h_layers+[recurrent_nodes]+[Y_d]
	print(D)

	X, n_frames, Y_, Y, loss, optimizer = RNN(D)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		e = 0
		best_loss = math.inf

		while True:
			e += 1
			epoch_loss = 0
			shuffle(titles)
			f = open(os.path.join(FFNN_models_path, 'output_log'+'.txt'), 'a')

			for title in titles:

				input_vector = load_input_title(title, normalized_input_features_path)
				length = len(input_vector)
				ouput_vector = load_output_title(title, normalized_output_features_path)
				print(title, length)

				_, current_loss = sess.run([optimizer, loss], {X:input_vector, n_frames:length, Y:ouput_vector})
				epoch_loss += current_loss

			print('Epoch:', e)
			print('Loss:', epoch_loss)

			f.write('Epoch:' + str(e) + '\n')
			f.write('Loss:' + str(epoch_loss) + '\n\n')

			if epoch_loss < best_loss:
				best_loss = epoch_loss

				saver = tf.train.Saver()
				saver.save(sess, os.path.join(FFNN_models_path, 'models'))

				print('Saving frozen graph...')

				graph = tf.get_default_graph()
				input_graph_def = graph.as_graph_def()

				output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, ['Y_']) 

				with tf.gfile.GFile(os.path.join(FFNN_models_path, 'frozen_model'), "wb") as f:
					f.write(output_graph_def.SerializeToString())

			if epoch_loss == 0.0:
				break
			
			if e == 5:
				break



