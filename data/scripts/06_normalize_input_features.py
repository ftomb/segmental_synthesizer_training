from multiprocessing import Pool
import numpy as np 
import json
import sys
import os

def load_titles(target_path, ext):
	titles = []
	for fn in os.listdir(target_path):
		basename, extension = os.path.splitext(fn)
		if extension == ext:
			titles.append(basename)
	return titles

def normalize(title, mean_std_path, input_features_path, normalized_input_features_path):

	with open(os.path.join(mean_std_path, 'input_mean_std.json'), 'r') as f:
		input_mean, input_std = json.load(f)

	with open(os.path.join(input_features_path, title+'.json'), 'r') as g:
		X = np.array(json.load(g))
		X -= input_mean
		X /= input_std+np.finfo(float).eps

	with open(os.path.join(normalized_input_features_path, title+'.json'), 'w') as h:
		json.dump(X.tolist(), h)


if __name__ == '__main__':

	mean_std_path = sys.argv[1]
	input_features_path = sys.argv[2]
	normalized_input_features_path = sys.argv[3]

	#mean_std_path = 'data/mean_std'
	#input_features_path = 'data/input_features'
	#normalized_input_features_path = 'data/normalized_input_features'

	input_titles = load_titles(input_features_path, '.json')

	p = Pool()
	p.starmap(normalize, [(title, mean_std_path, input_features_path, normalized_input_features_path) for title in input_titles])
