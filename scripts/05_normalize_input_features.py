from multiprocessing import Pool
import numpy as np 
import pickle
import sys
import os

def normalize(title, mean_std_path, input_features_path, normalized_input_features_path):

	with open(mean_std_path + 'input_mean_std.pickle' , "rb") as f: 
		input_mean, input_std = pickle.load(f)

	with open(input_features_path + title, "rb") as g: 
		X = pickle.load(g)
		X -= input_mean
		X /= input_std+np.finfo(float).eps

	with open(normalized_input_features_path + title, "wb") as h:
		pickle.dump(X, h)


if __name__ == '__main__':

	mean_std_path = sys.argv[1]
	input_features_path = sys.argv[2]
	normalized_input_features_path = sys.argv[3]

	input_titles = []
	for fn in os.listdir(input_features_path):
		basename, extension = os.path.splitext(fn)
		if extension == '.pickle':
			input_titles.append(fn)

	p = Pool()
	p.starmap(normalize, [(title, mean_std_path, input_features_path, normalized_input_features_path) for title in input_titles])
