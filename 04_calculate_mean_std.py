import numpy as np 
import pickle
import os


os.makedirs('mean_std')
output_titles = []
for fn in os.listdir('output_features/'):
	output_titles.append(fn[:-8])

def calculate_output_mean_std(output_titles):
	k = 0 
	M = 0
	S = 0
	for title in output_titles:
		with open('output_features/' + title + '.bap_mgc', "rb") as f: 
			X = pickle.load(f)
			for x in X:
				k += 1
				M_ = M + (x - M)/k
				S = S + (x - M)*(x - M_)
				M = M_
	return M, S/(k-1)

output_mean, output_var = calculate_output_mean_std(output_titles)
output_std = np.sqrt(output_var)

with open('mean_std/output_mean_std.pickle', "wb") as f:
	pickle.dump([output_mean, output_std], f)

input_titles = []
for fn in os.listdir('input_features/'):
	input_titles.append(fn[:-7])

def calculate_input_mean_std(input_titles):
	k = 0 
	M = 0
	S = 0
	for title in input_titles:
		with open('input_features/' + title + '.pickle', "rb") as f: 
			X = pickle.load(f)
			for x in X:
				k += 1
				M_ = M + (x - M)/k
				S = S + (x - M)*(x - M_)
				M = M_
	return M, S/(k-1)

input_mean, input_var = calculate_input_mean_std(input_titles)
input_std = np.sqrt(input_var)

with open('mean_std/input_mean_std.pickle', "wb") as f:
	pickle.dump([input_mean, input_std], f)




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