from multiprocessing import Pool
import subprocess
import sys
import os

def resample(title, input_path, output_path, extension):
	subprocess.call(['sox', os.path.join(input_path, title+extension), '-b', '16', os.path.join(output_path, title+extension), 'rate', str(48000)])

if __name__ == '__main__':

	input_path = sys.argv[1]
	output_path = sys.argv[2]
	
	wav_titles = []
	for fn in os.listdir(input_path):
		basename, extension = os.path.splitext(fn)
		if extension == '.wav':
			wav_titles.append(basename)

	p = Pool()
	p.starmap(resample, [(title, input_path, output_path, extension) for title in wav_titles])
