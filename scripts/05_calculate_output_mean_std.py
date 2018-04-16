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

def calculate_output_mean_std(output_titles, output_features_path):
	k = 0 
	M = 0
	S = 0
	for title in output_titles:
		with open(os.path.join(output_features_path, title + '.bap_mgc'), 'r') as f:
			X = np.array(json.load(f))
			for x in X:
				k += 1
				M_ = M + (x - M)/k
				S = S + (x - M)*(x - M_)
				M = M_
	return M, S/(k-1)


if __name__ == '__main__':

	output_features_path = sys.argv[1]
	mean_std_path = sys.argv[2]

	#output_features_path = 'data/output_features'
	#mean_std_path = 'data/mean_std'

	output_titles = load_titles(output_features_path, '.bap_mgc')

	output_mean, output_var = calculate_output_mean_std(output_titles, output_features_path)
	output_std = np.sqrt(output_var)

	with open(os.path.join(mean_std_path, 'output_mean_std.json'), 'w') as f:
		json.dump([output_mean.tolist(), output_std.tolist()], f)

