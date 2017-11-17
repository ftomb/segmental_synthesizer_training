from multiprocessing import Pool
import numpy as np 
import pickle
import tgt
import sys
import os


def process_phones(l):
	l_idx = [0]+[0 if l[i] == l[i-1] else 1 for i in range(1, len(l))]
	idxs = [0]+[i for i in range(0, len(l_idx)) if l_idx[i] == 1]+[len(l)]
	slices = [[idxs[i], idxs[i+1]] for i in range(0, len(idxs)-1)]
	l_phs = [l[s[0]:s[1]] for s in slices]
	l_ph_len = [len(i) for i in l_phs]
	l_ph_mid = [i[0] for i in l_phs]
	l_ph_bef = ['sil']+l_ph_mid[:-1]
	l_ph_bef_bef = ['sil']+l_ph_bef[:-1]
	l_ph_aft = l_ph_mid[1:]+['sil']
	l_ph_aft_aft = l_ph_aft[1:]+['sil']
	l_ph_befs = [l_ph_bef[i] for i in range(len(l_ph_len)) for j in range(l_ph_len[i])]
	l_ph_bef_befs = [l_ph_bef_bef[i] for i in range(len(l_ph_len)) for j in range(l_ph_len[i])]
	l_ph_afts = [l_ph_aft[i] for i in range(len(l_ph_len)) for j in range(l_ph_len[i])]
	l_ph_aft_afts = [l_ph_aft_aft[i] for i in range(len(l_ph_len)) for j in range(l_ph_len[i])]
	l_ph_lens = [[l_ph_len[i]] for i in range(len(l_ph_len)) for j in range(l_ph_len[i])]
	l_perc = [[j/len(l_phs[i])] for i in range(len(l_phs)) for j in range(len(l_phs[i]))]
	
	return l, l_ph_befs, l_ph_bef_befs, l_ph_afts, l_ph_aft_afts, l_perc, l_ph_lens

def extract_features(title, times_path, textgrid_path, phone_dict_path, f0_path, input_features_path):

	# get the times 
	with open(times_path + title + ".times", "rb") as f: 
		ts = pickle.load(f)

	tgname = textgrid_path + title + '.TextGrid'
	tg = tgt.read_textgrid(tgname)
	phones_tier = tg.get_tier_by_name('phones')

	phone_list = []
	for t in ts:
		try:
			phone_list.append(phones_tier.get_annotations_by_time(t)[0].text)
		except:
			phone_list.append('sil')

	ph, ph_b, ph_b_b, ph_a, ph_a_a, ph_perc, ph_len = process_phones(phone_list)

	with open(phone_dict_path + 'phone_dictionary.dict', "rb") as f:
		ph_dict = pickle.load(f)

	hot_ph = np.array([ph_dict[i] for i in ph])
	hot_ph_b = np.array([ph_dict[i] for i in ph_b])
	hot_ph_b_b = np.array([ph_dict[i] for i in ph_b_b])
	hot_ph_a = np.array([ph_dict[i] for i in ph_a])
	hot_ph_a_a = np.array([ph_dict[i] for i in ph_a_a])

	with open(f0_path + title + '.f0') as f:
		lf0 = np.log2([[float(l.strip())] for l in f], dtype=np.float64)

	input_vector = np.concatenate((hot_ph, hot_ph_b, hot_ph_b_b, hot_ph_a, hot_ph_a_a, ph_perc, ph_len), axis=1)

	if len(lf0) > len(input_vector):
		lf0 = lf0[:len(input_vector)]

	while len(lf0) < len(input_vector):
		lf0 = np.concatenate((lf0, [lf0[-1]]), axis=0)

	input_vector = np.concatenate((input_vector, lf0), axis=1)

	with open(input_features_path + title + ".pickle", "wb") as f:
		pickle.dump(input_vector, f)


if __name__ == '__main__':

	times_path = sys.argv[1]
	textgrid_path = sys.argv[2]
	phone_dict_path = sys.argv[3]
	f0_path = sys.argv[4]
	input_features_path = sys.argv[5]

	times_titles = []
	for fn in os.listdir(times_path):
		basename, extension = os.path.splitext(fn)
		if extension == '.times':
			times_titles.append(basename)

	textgrid_titles = []
	for fn in os.listdir(textgrid_path):
		basename, extension = os.path.splitext(fn)
		if extension == '.TextGrid':
			textgrid_titles.append(basename)

	f0_titles = []
	for fn in os.listdir(f0_path):
		basename, extension = os.path.splitext(fn)
		if extension == '.f0':
			f0_titles.append(basename)

	titles = set(times_titles).intersection(textgrid_titles).intersection(f0_titles)

	p = Pool()
	p.starmap(extract_features, [(title, times_path, textgrid_path, phone_dict_path, f0_path, input_features_path) for title in titles])
