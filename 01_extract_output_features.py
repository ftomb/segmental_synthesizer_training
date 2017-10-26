from scipy.io import wavfile
import pyworld as pw 
import numpy as np
import pysptk
import pickle
import os

fs = 48000

titles = []
for fn in os.listdir('wav_/'):
	titles.append(fn[:-4])

os.makedirs('output_features')
os.makedirs('extraction_times')

for title in titles:

	wav_stream = wavfile.read('wav_/' + title + '.wav')
	wav_stream = np.array(wav_stream[1], dtype=float)
	x = np.array(wav_stream, dtype=np.float64)

	_f0, ts = pw.dio(x, fs)    # raw pitch extractor

	f0 = pw.stonemask(x, _f0, ts, fs)  # pitch refinement
	sp = pw.cheaptrick(x, f0, ts, fs)  # extract smoothed spectrogram
	ap = pw.d4c(x, f0, ts, fs)         # extract aperiodicity

	bap = pysptk.sp2mc(ap, order=2, alpha=pysptk.util.mcepalpha(fs))
	mgc = pysptk.sp2mc(sp, order=60, alpha=pysptk.util.mcepalpha(fs))

	output_vector = np.concatenate((bap, mgc), axis=1)

	with open('output_features/' + title + ".bap_mgc", "wb") as f:
		pickle.dump(output_vector, f)

	with open('extraction_times/' + title + ".times", "wb") as g:
		pickle.dump(ts, g)
	
