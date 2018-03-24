from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import graph_util
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils
from tensorflow.python.ops import math_ops
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

def rnn_cell(recurrent_nodes):
	return tf.nn.rnn_cell.GRUCell(num_units=recurrent_nodes, activation=tf.nn.elu)

def rnn_layer(X, recurrent_nodes):
	rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=recurrent_nodes, activation=tf.nn.elu)
	return tf.nn.dynamic_rnn(cell=rnn_cell, inputs=X, dtype=tf.float32)

def bidirectional_rnn_layer(X, fw_cell, bw_cell, seq_len):
	return tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=X, dtype=tf.float32, sequence_length=seq_len)

def combine(L, W, B):
	return tf.add(tf.matmul(L, W), B)

def activate(L):
	alpha = 1.6732632423543772848170429916717
	scale = 1.0507009873554804934193349852946
	return scale*tf.where(L>=0.0, L, alpha*tf.nn.elu(L))

def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0, 
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))

def calculate_rmse(prediction, target):
	return tf.sqrt(tf.reduce_mean(tf.squared_difference(prediction, target)))
	
def momentum_optimizer(loss):

	# regularize loss by using L1
	l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
	weights = tf.trainable_variables()
	regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
	regularized_loss = loss + regularization_penalty 

	# regularize loss by using L2
	#regularized_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])*0.001

	# use momentum optimizer
	optimizer = tf.train.MomentumOptimizer(0.01, momentum=0.9, use_nesterov=True).minimize(regularized_loss)
	return optimizer


def RNN(D):

	X = tf.placeholder(tf.float32, [None, D[0]], name='X')
	dropoutRate = tf.placeholder(tf.float32, name='dropoutRate')
	is_training= tf.placeholder(tf.bool, name='is_training')

	W = [weight(D[i], D[i+1]) for i in range(len(D)-1)]
	B = [bias(D[i]) for i in range(1, len(D))]

	L = [X]
	for i in range(0, len(D)-3):
		l = activate(combine(L[-1], W[i], B[i]))
		dpl = dropout_selu(l, dropoutRate, training=is_training)
		L.append(dpl)

	n_frames = tf.placeholder(tf.int32, name='n_frames')
	seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
	L1 = tf.reshape(L[-1], [-1, n_frames, D[-3]])

	with tf.variable_scope('gru', initializer=tf.orthogonal_initializer(gain=tf.sqrt(2.0))):
		with tf.variable_scope('fw'):
			cell_fw = rnn_cell(D[-2])
		with tf.variable_scope('bw'):
			cell_bw = rnn_cell(D[-2])
		outputs, states = bidirectional_rnn_layer(L1, cell_fw, cell_bw, seq_len)

	outputs_concatenated = tf.concat(outputs, 2)
	outputs_reshaped = tf.reshape(outputs_concatenated, [-1, D[-2]*2])

	Y_ = tf.identity(combine(outputs_reshaped, weight(D[-2]*2, D[-1]), bias(D[-1])), name='Y_')
	Y = tf.placeholder(tf.float32, [None, D[-1]], name='Y')

	loss = calculate_rmse(Y_, Y)
	optimizer = momentum_optimizer(loss)

	return X, dropoutRate, is_training, n_frames, seq_len, Y_, Y, loss, optimizer

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
	forward_layers = 4
	h_layers = [forward_nodes for i in range(0, forward_layers)]
	D = [X_d]+h_layers+[recurrent_nodes]+[Y_d]
	print(D)

	X, dropoutRate, is_training, n_frames, seq_len, Y_, Y, loss, optimizer = RNN(D)

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
				_, current_loss = sess.run([optimizer, loss], {X:input_vector, dropoutRate:0.5, is_training:True, n_frames:length, seq_len:[length], Y:ouput_vector})
				print(title, length, current_loss)
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

			if e > 24
				break





