from scipy.interpolate import interp1d
from multiprocessing import Pool
from scipy.io import wavfile
import pyworld as pw 
import numpy as np
import pysptk
import pickle
import sys
import os

def interpolate(y):

	if y[0] == 0:
		for i, v in enumerate(y):
			if v != 0:
				y[0] = v
				break

	if y[-1] == 0:
		for i in reversed(range(len(y))):
			if y[i] != 0:
				y[-1] = y[i]
				break

	x = np.arange(len(y))
	idx = np.nonzero(y)
	interp = interp1d(x[idx],y[idx])
	
	return interp(x).reshape((len(interp(x)), 1))


def extract_features(title, wav_path, bap_mgc_path, times_path):
	
	fs = 48000
	wav_stream = wavfile.read(os.path.join(wav_path, title + '.wav'))
	wav_stream = np.array(wav_stream[1], dtype=float)
	x = np.array(wav_stream, dtype=np.float64)

	_f0, ts = pw.dio(x, fs)    # raw pitch extractor

	f0 = pw.stonemask(x, _f0, ts, fs)  # pitch refinement
	sp = pw.cheaptrick(x, f0, ts, fs)  # extract smoothed spectrogram
	ap = pw.d4c(x, f0, ts, fs)         # extract aperiodicity

	mgc = pysptk.sp2mc(sp, order=50, alpha=pysptk.util.mcepalpha(fs))
	bap = pysptk.sp2mc(ap, order=1, alpha=pysptk.util.mcepalpha(fs))
	bap = np.array([[j if abs(j-0) > 0.01 else 0 for j in i] for i in bap], dtype=np.float64)

	vuv = np.array([[0] if i[1] == 0 else [1] for i in bap], dtype=np.float64)
	bap = np.concatenate([interpolate(v.flatten()) for v in np.split(bap, 2, axis=1)], axis=1)

	output_vector = np.concatenate((vuv, bap, mgc), axis=1)

	with open(os.path.join(bap_mgc_path, title + ".bap_mgc"), "wb") as f:
		pickle.dump(output_vector, f)

	with open(os.path.join(times_path, title + ".times"), "wb") as g:
		pickle.dump(ts, g)
	
if __name__ == '__main__':

	wav_path = sys.argv[1]
	bap_mgc_path = sys.argv[2]
	times_path = sys.argv[3]

	titles = []
	for fn in os.listdir(wav_path):
		basename, extension = os.path.splitext(fn)
		if extension == '.wav':
			titles.append(basename)

	p = Pool()
	p.starmap(extract_features, [(title, wav_path, bap_mgc_path, times_path) for title in titles])
