import numpy as np 
import pickle
import os

with open('mean_std/' + 'output_mean_std.pickle' , "rb") as f: 
	output_mean, output_std = pickle.load(f)

output_titles = []
for fn in os.listdir('output_features/'):
	output_titles.append(fn)

os.makedirs('normalized_output_features')
for title in output_titles:

	with open('output_features/' + title, "rb") as g: 
		X = pickle.load(g)
		X -= output_mean
		X /= output_std

	with open('normalized_output_features/' + title, "wb") as h:
		pickle.dump(X, h)