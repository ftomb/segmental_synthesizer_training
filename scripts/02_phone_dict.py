import numpy as np
import pickle
import tgt
import sys
import os

def generate_phone_set(titles, textgrid_path):

	phone_list = []

	for title in titles:
		tgname = textgrid_path + title + '.TextGrid'

		tg = tgt.read_textgrid(tgname)
		phones_tier = tg.get_tier_by_name('phones')

		for phone in phones_tier:
			phone_list.append(phone.text)

	return set(phone_list)

def generate_phone_dict(phone_set):
	ph_dict = {}
	for j, v in enumerate(phone_set):
		ph_dict[v] = np.eye(len(phone_set), dtype=np.float64)[j]
	return ph_dict


if __name__ == '__main__':

	textgrid_path = sys.argv[1]
	phone_dict_path = sys.argv[2]

	titles = []
	for fn in os.listdir(textgrid_path):
		basename, extension = os.path.splitext(fn)
		if extension == '.TextGrid':
			titles.append(basename)
	
	phone_set = generate_phone_set(titles, textgrid_path)
	ph_dict = generate_phone_dict(phone_set)
	
	with open(phone_dict_path + 'phone_dictionary' + ".dict", "wb") as f:
		pickle.dump(ph_dict, f)

