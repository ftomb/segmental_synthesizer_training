from multiprocessing import Pool
import numpy as np 
import pickle
import sys
import os

def normalize(title, mean_std_path, output_features_path, normalized_output_features_path):

	with open(os.path.join(mean_std_path, 'output_mean_std.pickle'), "rb") as f:
		output_mean, output_std = pickle.load(f)

	with open(os.path.join(output_features_path, title), "rb") as g:
		X = pickle.load(g)
		X -= output_mean
		X /= output_std+np.finfo(float).eps

	with open(os.path.join(normalized_output_features_path, title), "wb") as h:
		pickle.dump(X, h)



if __name__ == '__main__':

	mean_std_path = sys.argv[1]
	output_features_path = sys.argv[2]
	normalized_output_features_path = sys.argv[3]


	output_titles = []
	for fn in os.listdir(output_features_path):
		basename, extension = os.path.splitext(fn)
		if extension == '.bap_mgc':
			output_titles.append(fn)

	p = Pool()
	p.starmap(normalize, [(title, mean_std_path, output_features_path, normalized_output_features_path) for title in output_titles])
