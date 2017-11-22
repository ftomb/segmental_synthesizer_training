from multiprocessing import Pool
import subprocess
import sys
import os

def extract_f0(title, input_path, output_path):
	subprocess.call(['praat/praat.exe', '--run', '00b_praat_script.praat', title, input_path, output_path])

if __name__ == '__main__':

	input_path = sys.argv[1]
	output_path = sys.argv[2]

	titles = []
	for fn in os.listdir(input_path):
		basename, extension = os.path.splitext(fn)
		if extension == '.wav':
			titles.append(basename)

	p = Pool()
	p.starmap(extract_f0, [(title, input_path, output_path) for title in titles])