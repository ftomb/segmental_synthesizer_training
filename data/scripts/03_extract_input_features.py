from multiprocessing import Pool
import numpy as np 
import json
import tgt
import sys
import os

def load_titles(target_path, ext):
	titles = []
	for fn in os.listdir(target_path):
		basename, extension = os.path.splitext(fn)
		if extension == ext:
			titles.append(basename)
	return titles

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

def extract_features(title, times_path, textgrid_path, phone_dict_path, f0_path, input_features_path):

	sil = 'sil'
	# get the times 
	with open(os.path.join(times_path, title + ".times"), "r") as f:
		ts = json.load(f)

	tgname = os.path.join(textgrid_path, title + '.TextGrid')
	tg = tgt.read_textgrid(tgname)
	tier_names = tg.get_tier_names()
	phones_tier_name = [name for name in tier_names if 'phones' in name][0]
	phones_tier = tg.get_tier_by_name(phones_tier_name)

	phone_list = []
	for t in ts:
		try:
			phone_list.append(phones_tier.get_annotations_by_time(t)[0].text.replace('sp', 'sil').replace('Q', '@@').replace('ts', 's'))
		except:
			phone_list.append(sil)

	ph, ph_b, ph_b_b, ph_a, ph_a_a, ph_perc, ph_len = process_phones(phone_list, sil)

	with open(os.path.join(phone_dict_path, 'phone_dictionary.dict'), "r") as f:
		ph_dict = json.load(f)

	hot_ph = np.array([ph_dict[i] for i in ph])
	hot_ph_b = np.array([ph_dict[i] for i in ph_b])
	hot_ph_b_b = np.array([ph_dict[i] for i in ph_b_b])
	hot_ph_a = np.array([ph_dict[i] for i in ph_a])
	hot_ph_a_a = np.array([ph_dict[i] for i in ph_a_a])

	with open(os.path.join(f0_path, title + '.f0'), "r") as f:
		f0s = json.load(f)
		f0 = [[f0] for f0 in f0s]
		lf0 = np.log2(f0, dtype=np.float64)

	input_vector = np.concatenate((hot_ph, hot_ph_b, hot_ph_b_b, hot_ph_a, hot_ph_a_a, ph_perc, np.divide(ph_len, 100), np.divide(lf0, 10)), axis=1)

	with open(os.path.join(input_features_path, title + ".json"), "w") as f:
		json.dump(input_vector.tolist(), f)



if __name__ == '__main__':

	times_path = sys.argv[1]
	textgrid_path = sys.argv[2]
	phone_dict_path = sys.argv[3]
	f0_path = sys.argv[4]
	input_features_path = sys.argv[5]

	#times_path = '../build/03_extraction_times'
	#textgrid_path = '../build/00_textgrid'
	#phone_dict_path = '../build/05_phone_dictionary'
	#f0_path = '../build/04_f0'
	#input_features_path = '../build/06_input_features'

	times_titles = load_titles(times_path, '.times')
	textgrid_titles = load_titles(textgrid_path, '.TextGrid')
	f0_titles = load_titles(f0_path, '.f0')

	titles = set(times_titles).intersection(textgrid_titles).intersection(f0_titles)

	p = Pool()
	p.starmap(extract_features, [(title, times_path, textgrid_path, phone_dict_path, f0_path, input_features_path) for title in titles])
