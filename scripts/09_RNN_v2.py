from tensorflow.python.framework import graph_util
from shutil import copyfile
from random import shuffle
import tensorflow as tf
import numpy as np
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
	init_range = tf.sqrt(12.0/(d1+d2))
	initializer = tf.random_uniform_initializer(minval=-init_range, maxval=init_range)
	return tf.Variable(initializer([d1, d2]))

def bias(d2):
	return tf.Variable(tf.zeros([d2]))

def rnn_cell(recurrent_nodes):
	return tf.nn.rnn_cell.GRUCell(num_units=recurrent_nodes, activation=tf.nn.elu)

def rnn_layer(X, cell):
	return tf.nn.dynamic_rnn(cell=cell, inputs=X, dtype=tf.float32)

def bidirectional_rnn_layer(X, fw_cell, bw_cell, seq_len):
	return tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=X, dtype=tf.float32, sequence_length=seq_len)

def combine(L, W, B):
	return tf.add(tf.matmul(L, W), B)

def calculate_rmse(prediction, target):
	return tf.sqrt(tf.reduce_mean(tf.squared_difference(prediction, target)))
	
def momentum_optimizer(loss):
	return tf.train.MomentumOptimizer(0.01, momentum=0.9, use_nesterov=True).minimize(loss)


def RNN(X_d, hidden_nodes, Y_d):

	X = tf.placeholder(tf.float32, [None, X_d], name='X')
	n_frames = tf.placeholder(tf.int32, name='n_frames')
	seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

	X_reshaped = tf.reshape(X, [-1, n_frames, X_d])

	# BiRNN: combine lingustic inputs backwards and forwards
	with tf.variable_scope('gru_1', initializer=tf.orthogonal_initializer(gain=tf.sqrt(2.0))):
		with tf.variable_scope('fw_1'):
			cell_fw_1 = rnn_cell(hidden_nodes)
		with tf.variable_scope('bw_1'):
			cell_bw_1 = rnn_cell(hidden_nodes)
		outputs1, states1 = bidirectional_rnn_layer(X_reshaped, cell_fw_1, cell_bw_1, seq_len)
	outputs1 = tf.concat(outputs1, 2)

	with tf.variable_scope('gru_2', initializer=tf.orthogonal_initializer(gain=tf.sqrt(2.0))):
		with tf.variable_scope('fw_2'):
			cell_fw_2 = rnn_cell(hidden_nodes*2)
		outputs2, states2 = rnn_layer(outputs1, cell_fw_2)
	outputs2 = tf.concat(outputs2, 2)
	outputs2_reshaped = tf.reshape(outputs2, [-1, hidden_nodes*2])

	W = weight(hidden_nodes*2, Y_d)
	B = bias(Y_d)
	Y_ = tf.identity(combine(outputs2_reshaped, W, B), name='Y_')
	Y = tf.placeholder(tf.float32, [None, Y_d], name='Y')

	loss = calculate_rmse(Y_, Y)
	optimizer = momentum_optimizer(loss)

	return X, n_frames, seq_len, Y_, Y, loss, optimizer

if __name__ == '__main__':

	split_train_test_valid_path = sys.argv[1]
	normalized_input_features_path = sys.argv[2]
	normalized_output_features_path = sys.argv[3]
	n_epochs = int(sys.argv[4])
	FFNN_models_path = sys.argv[5]
	frozen_models_path = sys.argv[6]

	#split_train_test_valid_path = '../build/11_split_test_train_valid'
	#normalized_input_features_path = '../build/09_normalized_input_features'
	#normalized_output_features_path = '../build/10_normalized_output_features'
	#n_epochs = 3
	#FFNN_models_path = '../build/12_FFNN_models'
	#frozen_models_path = '../build/13_frozen_models'

	with open(os.path.join(split_train_test_valid_path, 'split_titles.json')) as f:    
		titles_json = json.load(f)
	train_titles = titles_json['train']
	#train_titles = train_titles[:1]
	print(train_titles)

	input_titles = load_titles(normalized_input_features_path, '.json')
	output_titles = load_titles(normalized_output_features_path, '.bap_mgc')
	titles = sorted(set(input_titles).intersection(output_titles).intersection(train_titles))

	X_d = len(load_input_title(titles[0], normalized_input_features_path)[0])
	Y_d = len(load_output_title(titles[0], normalized_output_features_path)[0])

	# Define dimensions of the neural network

	hidden_nodes = 256
	print(X_d, hidden_nodes, Y_d)

	X, n_frames, seq_len, Y_, Y, loss, optimizer = RNN(X_d, hidden_nodes, Y_d)

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

				_, current_loss = sess.run([optimizer, loss], {X:input_vector, n_frames:length, seq_len:[length], Y:ouput_vector})
				epoch_loss += current_loss

			print('Epoch:', e)
			print('Loss:', epoch_loss)

			f.write('Epoch:' + str(e) + '\n')
			f.write('Loss:' + str(epoch_loss) + '\n\n')

			if e>0:

				saver = tf.train.Saver()
				saver.save(sess, os.path.join(FFNN_models_path, 'models'))

				print('Saving frozen graph...')

				graph = tf.get_default_graph()
				input_graph_def = graph.as_graph_def()

				output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, ['Y_']) 

				with tf.gfile.GFile(os.path.join(FFNN_models_path, 'frozen_model'+'_'+str(e)), "wb") as f:
					f.write(output_graph_def.SerializeToString())

				copyfile(os.path.join(FFNN_models_path,'frozen_model'+'_'+str(e)), os.path.join(frozen_models_path,'frozen_model'+'_'+str(e)))				

			if e >= n_epochs:
				break

