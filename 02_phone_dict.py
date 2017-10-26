import numpy as np
import pickle
import tgt
import os

titles = []
for fn in os.listdir('wav_/'):
	titles.append(fn[:-4])

phone_list = []

for title in titles:

	tgname = 'textgrid/' + title + '.TextGrid'
	try:
		tg = tgt.read_textgrid(tgname)
		phones_tier = tg.get_tier_by_name('phones')

		for phone in phones_tier:
			phone_list.append(phone.text)
	except:
		pass

ph_dict = {}
for j, v in enumerate(set(phone_list)):
	ph_dict[v] = np.eye(len(set(phone_list)), dtype=np.float64)[j]

os.makedirs('phone_dictionary')

with open('phone_dictionary/' + 'phone_dictionary' + ".dict", "wb") as f:
	pickle.dump(ph_dict, f)
