import numpy as np 
import pickle
import os


with open('mean_std/' + 'input_mean_std.pickle' , "rb") as f: 
	input_mean, input_std = pickle.load(f)

input_titles = []
for fn in os.listdir('input_features/'):
	input_titles.append(fn)

os.makedirs('normalized_input_features')
for title in input_titles:

	with open('input_features/' + title, "rb") as g: 
		X = pickle.load(g)
		X -= input_mean
		X /= input_std

	with open('normalized_input_features/' + title, "wb") as h:
		pickle.dump(X, h)
