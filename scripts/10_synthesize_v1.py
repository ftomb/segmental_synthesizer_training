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

def load_f0(title, f0_path):
	# use this when you have time column and an f0 column
	with open(os.path.join(f0_path, title+'.f0')) as f:
		return np.array([float(l.strip().split()[1]) for l in f], dtype=np.float64)

#def load_f0(title, f0_path):
	# use this for f0s stored as json without timestamps 
	#with open(os.path.join(f0_path, title+'.f0')) as f:
		#f0 = json.load(f)
		#return np.array(f0, dtype=np.float64)

def load_merlin_f0(title, f0_path):
	with open(os.path.join(f0_path, title+'.f0'), 'r') as f:
		return [np.exp(float(l.strip())) if float(l.strip()) > 0 else 0 for l in f]
		
def process_phones(l, sil):
	l_idx = [0]+[0 if l[i] == l[i-1] else 1 for i in range(1, len(l))]
	idxs = [0]+[i for i in range(0, len(l_idx)) if l_idx[i] == 1]+[len(l)]
	slices = [[idxs[i], idxs[i+1]] for i in range(0, len(idxs)-1)]
	l_phs = [l[s[0]:s[1]] for s in slices]
	l_ph_len = [len(i) for i in l_phs]
	l_ph_mid = [i[0] for i in l_phs]
	l_ph_bef = [sil]+l_ph_mid[:-1]
	l_ph_bef_bef = [sil]+l_ph_bef[:-1]
	l_ph_aft = l_ph_mid[1:]+[sil]
	l_ph_aft_aft = l_ph_aft[1:]+[sil]
	l_ph_befs = [l_ph_bef[i] for i in range(len(l_ph_len)) for j in range(l_ph_len[i])]
	l_ph_bef_befs = [l_ph_bef_bef[i] for i in range(len(l_ph_len)) for j in range(l_ph_len[i])]
	l_ph_afts = [l_ph_aft[i] for i in range(len(l_ph_len)) for j in range(l_ph_len[i])]
	l_ph_aft_afts = [l_ph_aft_aft[i] for i in range(len(l_ph_len)) for j in range(l_ph_len[i])]
	l_ph_lens = [[l_ph_len[i]] for i in range(len(l_ph_len)) for j in range(l_ph_len[i])]
	l_perc = [[j/len(l_phs[i])] for i in range(len(l_phs)) for j in range(len(l_phs[i]))]
	
	return l, l_ph_befs, l_ph_bef_befs, l_ph_afts, l_ph_aft_afts, l_perc, l_ph_lens


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


def synthesize_title(title, textgrid_path, f0_path, model_path, output_path):

	# Load NN Model 
	print('Loading Model...')

	with tf.gfile.GFile(os.path.join(model_path, 'frozen_model'), "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	with tf.Graph().as_default() as graph:
		tf.import_graph_def(graph_def, input_map=None, return_elements=None, name=None, op_dict=None, producer_op_list=None)

	X = graph.get_tensor_by_name('import/X:0')
	n_frames = graph.get_tensor_by_name('import/n_frames:0')
	Y_ = graph.get_tensor_by_name('import/Y_:0')


	print(title)
	print('Preparing input...')
	# Load f0
	sil = 'sil'

	f0 = load_f0(title, f0_path)

	# This is needed by the synthesizer
	lf0 = np.log2([[float(i)] for i in f0], dtype=np.float64)

	# Generate time points for phone extraction
	ts = [i*0.005 for i in range(len(lf0))]

	# Load TextGrid
	tgname = os.path.join(textgrid_path, title + '.TextGrid')
	tg = tgt.read_textgrid(tgname)
	tier_names = tg.get_tier_names()
	phones_tier_name = [name for name in tier_names if 'phones' in name][0]
	phones_tier = tg.get_tier_by_name(phones_tier_name)

	# Generate phone list
	phone_list = []

	for t in ts:
		try:
			phone_list.append(phones_tier.get_annotations_by_time(t)[0].text.replace('Q', '@@').replace('ts', 's').replace('sp', 'sil'))
		except:
			phone_list.append(sil)

	# generate phone vectors
	ph, ph_b, ph_b_b, ph_a, ph_a_a, ph_perc, ph_len = process_phones(phone_list, sil)

	with open(os.path.join(model_path, 'phone_dictionary.dict'), "r") as f:
		ph_dict = json.load(f)

	# Convert phone vectors to hot vectors
	hot_ph = np.array([ph_dict[i] for i in ph])
	hot_ph_b = np.array([ph_dict[i] for i in ph_b])
	hot_ph_b_b = np.array([ph_dict[i] for i in ph_b_b])
	hot_ph_a = np.array([ph_dict[i] for i in ph_a])
	hot_ph_a_a = np.array([ph_dict[i] for i in ph_a_a])


	# Concatenate all the input vectors
	input_vector = np.concatenate((hot_ph, hot_ph_b, hot_ph_b_b, hot_ph_a, hot_ph_a_a, ph_perc, np.divide(ph_len, 100), np.divide(lf0, 10)), axis=1)

	with open(os.path.join(model_path, 'input_mean_std.json'), "r") as f:
		input_mean, input_std = json.load(f)

	input_mean = np.array(input_mean)
	input_std = np.array(input_std)

	# Normalize input vector
	print('Normalizing...')
	input_vector -= input_mean
	input_vector /= input_std+np.finfo(float).eps

	# Inference
	print('Doing inference...')
	length = len(input_vector)
	with tf.Session(graph=graph) as sess:
		output_vector = sess.run([Y_], {X:input_vector, n_frames:length})
	prediction = output_vector[0]


	print('Denormalizing...')
	with open(os.path.join(model_path, 'output_mean_std.json'), "r") as g:
		output_mean, output_std = json.load(g)

	output_mean = np.array(output_mean)
	output_std = np.array(output_std)

	prediction *= output_std
	prediction += output_mean

	if len(lf0) > len(prediction):
		lf0 = lf0[:len(prediction)]
	else:
		prediction = prediction[:len(lf0)]

	fs = 48000

	vuv = np.array(prediction[:,0:1], dtype=np.float64)
	vuv = np.array([[1] if i[0] > 0.5 else [0] for i in vuv])

	bap = np.array(prediction[:,1:7], dtype=np.float64)
	bap = np.concatenate([(col.flatten()*vuv.flatten()).reshape((-1, 1)) for col in np.split(bap, len(bap[0]), axis=1)], axis=1)

	mgc = np.array(prediction[:,7:], dtype=np.float64)
	mgc = merlin_post_filter(mgc, alpha=pysptk.util.mcepalpha(fs))

	f0 = f0.flatten()*vuv.flatten()

	sp = pysptk.mc2sp(mgc, fftlen=2048, alpha=pysptk.util.mcepalpha(fs))
	ap = pysptk.mc2sp(bap, fftlen=2048, alpha=pysptk.util.mcepalpha(fs))

	y = pw.synthesize(f0, sp, ap, fs)
	wavfile.write(os.path.join(output_path, title + '.wav'), fs, np.array(y, dtype=np.int16))



if __name__ == '__main__':

	textgrid_path = sys.argv[1]
	f0_path = sys.argv[2]
	model_path = sys.argv[3]
	output_path = sys.argv[4]

	#textgrid_path = '../build/textgrid'
	#f0_path = '../build/f0'
	#model_path = '../build/model'
	#output_path = '../build/wav'

	textgrid_titles = load_titles(textgrid_path, '.TextGrid')
	f0_titles = load_titles(f0_path, '.f0')

	titles = list(set(textgrid_titles).intersection(f0_titles))

	if titles == []:
		sys.exit('No input files found! Please provide .f0 and .TextGrid files!')

	p = Pool()
	p.starmap(synthesize_title, [(title, textgrid_path, f0_path, model_path, output_path) for title in titles])



	 
