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

def normalize(title, mean_std_path, output_features_path, normalized_output_features_path):

	with open(os.path.join(mean_std_path, 'output_mean_std.json'), 'r') as f:
		output_mean, output_std = json.load(f)

	with open(os.path.join(output_features_path, title+'.bap_mgc'), 'r') as g:
		X = np.array(json.load(g))
		X -= output_mean
		X /= output_std+np.finfo(float).eps
		

	with open(os.path.join(normalized_output_features_path, title+'.bap_mgc'), 'w') as h:
		json.dump(X.tolist(), h)



if __name__ == '__main__':

	mean_std_path = sys.argv[1]
	output_features_path = sys.argv[2]
	normalized_output_features_path = sys.argv[3]

	#mean_std_path = 'data/mean_std'
	#output_features_path = 'data/output_features'
	#normalized_output_features_path = 'data/normalized_output_features'

	output_titles = load_titles(output_features_path, '.bap_mgc')

	p = Pool()
	p.starmap(normalize, [(title, mean_std_path, output_features_path, normalized_output_features_path) for title in output_titles])
