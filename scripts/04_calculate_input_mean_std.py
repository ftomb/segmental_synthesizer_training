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

def calculate_input_mean_std(input_titles, input_features_path):
	k = 0 
	M = 0
	S = 0
	for title in input_titles:
		with open(os.path.join(input_features_path, title + '.json'), 'r') as f:
			X = np.array(json.load(f))

			for x in X:
				k += 1
				M_ = M + (x - M)/k
				S = S + (x - M)*(x - M_)
				M = M_
	return M, S/(k-1)

if __name__ == '__main__':

	input_features_path = sys.argv[1]
	mean_std_path = sys.argv[2]

	#input_features_path = 'data/input_features'
	#output_features_path = 'data/output_features'
	#mean_std_path = 'data/mean_std'

	input_titles = load_titles(input_features_path, '.json')
	
	input_mean, input_var = calculate_input_mean_std(input_titles, input_features_path)
	input_std = np.sqrt(input_var)

	with open(os.path.join(mean_std_path, 'input_mean_std.json'), 'w') as f:
		json.dump([input_mean.tolist(), input_std.tolist()], f)




'''
#this is to debug the online mean and std algorithm, here we compare the results I get above with the built-in numpy function

mega = []
for title in output_titles:
	with open('input_features/' + title + '.pickle', "rb") as f: 
		X = pickle.load(f)
		for x in X:
			mega.append(x)
print('mean')
print(np.mean(mega, axis=0))
print('std')
print(np.std(mega, axis=0, ddof=1))

'''

