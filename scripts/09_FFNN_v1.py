from tensorflow.python.framework import graph_util
from shutil import copyfile
from random import shuffle
import tensorflow as tf
import numpy as np
import numbers
import json
import math
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def load_titles(target_path, extension):
    titles = []
    for fn in os.listdir(target_path):
        basename, ext = os.path.splitext(fn)
        if ext == extension:
            titles.append(basename)
    return titles

def load_input_title(title, normalized_input_features_path):
	with open(os.path.join(normalized_input_features_path, title + '.json'), "r") as g:
		return json.load(g)

def load_output_title(title, normalized_output_features_path):
	with open(os.path.join(normalized_output_features_path, title + '.bap_mgc'), "r") as g:
		return json.load(g)

def weight(d1, d2):
	return tf.Variable(tf.random_normal([d1, d2],stddev=np.sqrt(1/d1)))

def bias(d2):
	return tf.Variable(tf.random_normal([d2], stddev=0))

def rnn_layer(X, recurrent_nodes):
	rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=recurrent_nodes, activation=tf.nn.elu)
	return tf.nn.dynamic_rnn(cell=rnn_cell, inputs=X, dtype=tf.float32)

def combine(L, W, B):
	return tf.add(tf.matmul(L, W), B)

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
	B = [bias(D[i]) for i in range(1, len(D))]

	L = [X]
	for i in range(0, len(D)-3):
		L.append(activate(combine(L[-1], W[i], B[i])))

	n_frames = tf.placeholder(tf.int32, name='n_frames')
	L1 = tf.reshape(L[-1], [-1, n_frames, D[-3]])
	outputs, states = rnn_layer(L1, D[-2])
	outputs = tf.reshape(outputs, [-1, D[-2]])

	Y_ = tf.identity(combine(outputs, W[-1], B[-1]), name='Y_')
	Y = tf.placeholder(tf.float32, [None, D[-1]], name='Y')

	loss = calculate_rmse(Y_, Y)
	optimizer = momentum_optimizer(loss)

	return X, n_frames, Y_, Y, loss, optimizer

if __name__ == '__main__':

	split_train_test_valid_path = sys.argv[1]
	normalized_input_features_path = sys.argv[2]
	normalized_output_features_path = sys.argv[3]
	FFNN_models_path = sys.argv[4]
	frozen_models_path = sys.argv[5]

	#split_train_test_valid_path = '11_split_train_test_valid'
	#normalized_input_features_path = '09_normalized_input_features'
	#normalized_output_features_path = '10_normalized_output_features'
	#FFNN_models_path = 'FFNN_models'
	#frozen_models_path = 'frozen_models'

	with open(os.path.join(split_train_test_valid_path, 'split_titles.json')) as f:    
		titles_json = json.load(f)
	train_titles = titles_json['train']


	input_titles = load_titles(normalized_input_features_path, '.json')
	output_titles = load_titles(normalized_output_features_path, '.bap_mgc')
	titles = sorted(set(input_titles).intersection(output_titles).intersection(train_titles))

	X_d = len(load_input_title(titles[0], normalized_input_features_path)[0])
	Y_d = len(load_output_title(titles[0], normalized_output_features_path)[0])

	# Define dimensions of the neural network

	forward_nodes = 512
	recurrent_nodes = forward_nodes
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

			saver = tf.train.Saver()
			saver.save(sess, os.path.join(FFNN_models_path, 'models'))

			print('Saving frozen graph...')

			graph = tf.get_default_graph()
			input_graph_def = graph.as_graph_def()

			output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, ['Y_']) 

			with tf.gfile.GFile(os.path.join(FFNN_models_path, 'frozen_model'+'_'+str(e)), "wb") as f:
				f.write(output_graph_def.SerializeToString())

			copyfile(os.path.join(FFNN_models_path,'frozen_model'+'_'+str(e)), os.path.join(frozen_models_path,'frozen_model'+'_'+str(e)))				

			if e > 100:
				break





