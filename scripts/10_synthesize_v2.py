from tensorflow.python.framework import graph_util
from multiprocessing import Pool
from scipy.io import wavfile
import tensorflow as tf
import pyworld as pw
import numpy as np
import subprocess
import pysptk
import json
import math
import tgt
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

def load_lf0(title, path):
	with open(os.path.join(path, title+'.lf0')) as f:
		return json.load(f)

def load_json(title, path):
	with open(os.path.join(path, title+'.json')) as f:
		return json.load(f)

def load_output(title, path):
	with open(os.path.join(path, title+'.bap_mgc')) as f:
		return json.load(f)

def merlin_post_filter(mgc, alpha,
                       minimum_phase_order=511, fftlen=2048,
                       coef=1.4, weight=None):

    _, D = mgc.shape
    if weight is None:
        weight = np.ones(D) * coef
        weight[:2] = 1
    assert len(weight) == D

    mgc_r0 = pysptk.c2acr(pysptk.freqt(
        mgc, minimum_phase_order, alpha=-alpha), 0, fftlen).flatten()
    mgc_p_r0 = pysptk.c2acr(pysptk.freqt(
        mgc * weight, minimum_phase_order, -alpha), 0, fftlen).flatten()
    mgc_b0 = pysptk.mc2b(weight * mgc, alpha)[:, 0]
    mgc_p_b0 = np.log(mgc_r0 / mgc_p_r0) / 2 + mgc_b0
    mgc_p_mgc = pysptk.b2mc(
        np.hstack((mgc_p_b0[:, None], pysptk.mc2b(mgc * weight, alpha)[:, 1:])), alpha)

    return mgc_p_mgc

def synthesize_title(title, lf0_path, input_features_path, model_path, output_mean_path, n_epochs, output_path, delta, v2):

	# Load NN Model 
	print('Loading Model...')

	with tf.gfile.GFile(os.path.join(model_path, 'frozen_model_'+n_epochs), "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	with tf.Graph().as_default() as graph:
		tf.import_graph_def(graph_def, input_map=None, return_elements=None, name=None, op_dict=None, producer_op_list=None)

	X = graph.get_tensor_by_name('import/X:0')
	n_frames = graph.get_tensor_by_name('import/n_frames:0')
	if v2 == True:
		seq_len = graph.get_tensor_by_name('import/seq_len:0')
	else:
		seq_len = None
	Y_ = graph.get_tensor_by_name('import/Y_:0')

	lf0 = np.array(load_lf0(title, lf0_path), dtype=np.float64)
	input_vector = load_json(title, input_features_path)

	# Inference
	print('Doing inference...')
	length = len(input_vector)
	with tf.Session(graph=graph) as sess:
		if v2 == True:
			output_vector = sess.run([Y_], {X:input_vector, n_frames:length, seq_len:[length]})
		else:
			output_vector = sess.run([Y_], {X:input_vector, n_frames:length})
	prediction = output_vector[0]

	print('Denormalizing...')
	with open(os.path.join(output_mean_path, 'output_mean_std.json'), "r") as g:
		output_mean, output_std = json.load(g)

	output_mean = np.array(output_mean)
	output_std = np.array(output_std)

	prediction *= output_std
	prediction += output_mean


	fs = 48000

	vuv = np.array(prediction[:,0:1], dtype=np.float64)
	vuv = np.array([[1] if i[0] > 0.5 else [0] for i in vuv])

	f0 = np.exp2(lf0[:,0:1], dtype=np.float64)
	f0 = f0.flatten()*vuv.flatten()

	if delta==True:
		# dunno what to do with the deltas, so for now just extract the static values and ignore deltas
		bap = np.array(prediction[:,1:7], dtype=np.float64)
		mgc = np.array(prediction[:,19:80], dtype=np.float64)

	else:
		bap = np.array(prediction[:,1:7], dtype=np.float64)
		mgc = np.array(prediction[:,7:], dtype=np.float64)

	bap = np.concatenate([(col.flatten()*vuv.flatten()).reshape((-1, 1)) for col in np.split(bap, len(bap[0]), axis=1)], axis=1)
	mgc = merlin_post_filter(mgc, alpha=pysptk.util.mcepalpha(fs))

	sp = pysptk.mc2sp(mgc, fftlen=2048, alpha=pysptk.util.mcepalpha(fs))
	ap = pysptk.mc2sp(bap, fftlen=2048, alpha=pysptk.util.mcepalpha(fs))

	y = pw.synthesize(f0, sp, ap, fs)
	wavfile.write(os.path.join(output_path, title+'_'+n_epochs+'.wav'), fs, np.array(y, dtype=np.int16))



if __name__ == '__main__':

	input_features_path = sys.argv[1]
	lf0_path = sys.argv[2]
	model_path = sys.argv[3]
	output_mean_path = sys.argv[4]
	n_epochs = sys.argv[5]
	split_train_test_valid_path = sys.argv[6]
	output_path = sys.argv[7]

	#input_features_path = '../build/09_normalized_input_features'
	#lf0_path = '../build/04_lf0'
	#output_features_path = '../build/10_normalized_output_features'
	#model_path = '../build/13_frozen_models'
	#output_mean_path = '../build/08_output_mean_std'
	#output_path = '../build/14_wav'

	#textgrid_path = '../build/00_textgrid'
	#f0_path = '../build/04_f0'
	#model_path = '../build/model'
	#output_path = '../build/14_wav'

	with open(os.path.join(split_train_test_valid_path, 'split_titles.json')) as f:    
		titles_json = json.load(f)
	valid_titles = titles_json['valid']
	#valid_titles = valid_titles[:1]
	print(valid_titles)


	input_titles = load_titles(input_features_path, '.json')

	titles = list(set(input_titles).intersection(valid_titles))

	# True if you included delta and delta-delta features
	delta = True

	# True if you trained the v2 model (birnn+rnn), False if you trained the v1 model(ffnn+rnn)
	v2 = True

	if titles == []:
		sys.exit('No input files found! Please provide .f0 and .TextGrid files!')

 
	p = Pool()
	p.starmap(synthesize_title, [(title, lf0_path, input_features_path, model_path, output_mean_path, n_epochs, output_path, delta, v2) for title in titles])



	 
