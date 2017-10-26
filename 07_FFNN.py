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

def load_output_mean_std():
	with open('mean_std/' + 'output_mean_std' + '.pickle', "rb") as g: 
		return pickle.load(g)

def load_f0(title):
	with open('f0/'+title+'.f0') as f:
		return np.array([float(l.strip()) for l in f], dtype=np.float64)

def weight(d1, d2):
	return tf.Variable(tf.random_normal([d1, d2],stddev=np.sqrt(1/d1)))

def bias(d2):
	return tf.Variable(tf.random_normal([d2], stddev=0))

def combine(L, W, B):
	return tf.add(tf.matmul(L, W), B)

def selu(x):
	alpha = 1.6732632423543772848170429916717
	scale = 1.0507009873554804934193349852946
	return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def activate(x):
    return selu(x)

def calculate_rmse(prediction, target):
	return tf.sqrt(tf.reduce_mean(tf.squared_difference(prediction, target)))

def momentum_optimizer(loss):
	return tf.train.MomentumOptimizer(0.01, momentum=0.9, use_nesterov=True).minimize(loss)

def FFNN(D):

	X = tf.placeholder(tf.float32, [None, D[0]])
	W = [weight(D[i], D[i+1]) for i in range(len(D)-1)]
	B = [bias(D[i]) for i in range(1, len(D))]

	L = []
	L_ = X
	for i in range(0, len(D)-2):
		_L = activate(combine(L_, W[i], B[i]))
		L.append(_L)
		L_ = _L

	Y_ = tf.identity(combine(L[-1], W[-1], B[-1]), name='Y_')
	Y = tf.placeholder(tf.float32, [None, D[-1]])

	loss = calculate_rmse(Y_, Y)
	optimizer = momentum_optimizer(loss)

	return X, Y_, Y, loss, optimizer

titles = load_titles()

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

os.makedirs('audio_outputs')
from scipy.io import wavfile
import pyworld as pw
import pysptk
fs = 48000
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	e = 0
	best_loss = math.inf
	while True:
		e += 1
		epoch_loss = 0

		for t, title in enumerate(titles[30:]):
			print(title)

			input_vector = load_input_title(title)
			ouput_vector = load_output_title(title)
			print(len(input_vector))

			_, current_loss = sess.run([optimizer, loss], {X:input_vector, Y:ouput_vector})
			epoch_loss += current_loss


			title = titles[0]

			input_vector = load_input_title(title)
			ouput_vector = load_output_title(title)

			output_mean, output_std = load_output_mean_std()

			prediction = sess.run([Y_], {X:input_vector})
			prediction = prediction[0]
			
			prediction *= output_std
			prediction += output_mean


			bap = np.array(prediction[:,0:3], dtype=np.float64)
			bap = np.array([[0.0 if val>-0.1 else val for val in row] for row in bap], dtype=np.float64)
			mgc = np.array(prediction[:,3:], dtype=np.float64)

			f0 = load_f0(title)

			sp = pysptk.mc2sp(mgc, fftlen=1024, alpha=pysptk.util.mcepalpha(fs))
			ap = pysptk.mc2sp(bap, fftlen=1024, alpha=pysptk.util.mcepalpha(fs))

			y = pw.synthesize(f0, sp, ap, fs)
			wavfile.write('audio_outputs/' + str(e) +str(t) +  'a.wav', fs, np.array(y, dtype=np.int16))

			title = titles[2]


			input_vector = load_input_title(title)
			ouput_vector = load_output_title(title)

			output_mean, output_std = load_output_mean_std()

			prediction = sess.run([Y_], {X:input_vector})
			prediction = prediction[0]
			
			prediction *= output_std
			prediction += output_mean


			bap = np.array(prediction[:,0:3], dtype=np.float64)
			bap = np.array([[0.0 if val>-0.1 else val for val in row] for row in bap], dtype=np.float64)
			mgc = np.array(prediction[:,3:], dtype=np.float64)

			f0 = load_f0(title)

			sp = pysptk.mc2sp(mgc, fftlen=1024, alpha=pysptk.util.mcepalpha(fs))
			ap = pysptk.mc2sp(bap, fftlen=1024, alpha=pysptk.util.mcepalpha(fs))

			y = pw.synthesize(f0, sp, ap, fs)
			wavfile.write('audio_outputs/' + str(e) +str(t) +'b.wav', fs, np.array(y, dtype=np.int16))


			title = titles[7]

			input_vector = load_input_title(title)
			ouput_vector = load_output_title(title)

			output_mean, output_std = load_output_mean_std()

			prediction = sess.run([Y_], {X:input_vector})
			prediction = prediction[0]
			
			prediction *= output_std
			prediction += output_mean


			bap = np.array(prediction[:,0:3], dtype=np.float64)
			bap = np.array([[0.0 if val>-0.1 else val for val in row] for row in bap], dtype=np.float64)
			mgc = np.array(prediction[:,3:], dtype=np.float64)

			f0 = load_f0(title)

			sp = pysptk.mc2sp(mgc, fftlen=1024, alpha=pysptk.util.mcepalpha(fs))
			ap = pysptk.mc2sp(bap, fftlen=1024, alpha=pysptk.util.mcepalpha(fs))

			y = pw.synthesize(f0, sp, ap, fs)
			wavfile.write('audio_outputs/' + str(e) + str(t) + 'c.wav', fs, np.array(y, dtype=np.int16))

		print('Epoch:', e)
		print('Loss:', epoch_loss)
		'''
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
		'''

