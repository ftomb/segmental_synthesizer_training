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


def generate_phone_set(titles, textgrid_path):

	phone_list = {}

	for title in titles:
		tgname = os.path.join(textgrid_path, title + '.TextGrid')

		tg = tgt.read_textgrid(tgname)
		tier_names = tg.get_tier_names()
		phones_tier_name = [name for name in tier_names if 'phones' in name][0]
		phones_tier = tg.get_tier_by_name(phones_tier_name)
		
		for phone in phones_tier:
			ph = phone.text.replace('sp', 'sil').replace('Q', '@@').replace('ts', 's')
			if ph not in phone_list.keys():
				phone_list[ph] = 1
			else:
				phone_list[ph] += 1


	return sorted(list(phone_list.keys()))

def generate_phone_dict(phone_set):
	ph_dict = {}
	for j, v in enumerate(phone_set):
		ph_dict[v] = np.eye(len(phone_set), dtype=np.int16)[j].tolist()
	return ph_dict


if __name__ == '__main__':

	textgrid_path = sys.argv[1]
	phone_dict_path = sys.argv[2]

	#textgrid_path = 'data/textgrid'
	#phone_dict_path = 'data/dictionaries'

	titles = load_titles(textgrid_path, '.TextGrid')
	phone_set = generate_phone_set(titles, textgrid_path)
	print(phone_set)

	ph_dict = generate_phone_dict(phone_set)
	
	with open(os.path.join(phone_dict_path, 'phone_dictionary' + ".dict"), "w") as f:
		json.dump(ph_dict, f)
