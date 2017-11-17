import numpy as np 
import pickle
import sys
import os

def calculate_output_mean_std(output_titles, output_features_path):
	k = 0 
	M = 0
	S = 0
	for title in output_titles:
		with open(output_features_path + title + '.bap_mgc', "rb") as f: 
			X = pickle.load(f)
			for x in X:
				k += 1
				M_ = M + (x - M)/k
				S = S + (x - M)*(x - M_)
				M = M_
	return M, S/(k-1)


if __name__ == '__main__':

	output_features_path = sys.argv[1]
	mean_std_path = sys.argv[2]

	output_titles = []
	for fn in os.listdir(output_features_path):
		basename, extension = os.path.splitext(fn)
		if extension == '.bap_mgc':
			output_titles.append(basename)

	output_mean, output_var = calculate_output_mean_std(output_titles, output_features_path)
	output_std = np.sqrt(output_var)

	with open(mean_std_path+'output_mean_std.pickle', "wb") as f:
		pickle.dump([output_mean, output_std], f)