from scipy import interpolate as interp
from scipy.interpolate import interp1d
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import pyworld as pw 
import numpy as np
import pysptk
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

def interpolate_f0(f0s):

	f0s = [np.log2(float(f0)) if f0 > 0 else f0 for f0 in f0s]

	org = f0s
	f0s_len = len(f0s)

	times_f0s = [[i*0.005, f0] for i, f0 in enumerate(f0s)]

	start_time = times_f0s[0][0]
	end_time = times_f0s[-1][0]

	no_zeros = [pair for pair in times_f0s if pair[1] > 0.0]
	f0_timepoints = [start_time]+[pair[0] for pair in no_zeros]+[end_time]

	f0s =[pair[1] for pair in no_zeros]
	f0s = [f0s[0]]+f0s+[f0s[-1]]

	f = interp.interp1d(f0_timepoints, f0s, kind='linear')
	x = np.linspace(start_time, end_time, f0s_len)
	return np.array([[f0] for f0 in f(x)], dtype=np.float64)

def interpolate_bap(baps):

	baps = [float(bap) for bap in baps]
	org = baps
	baps_len = len(baps)

	times_baps = [[i*0.005, bap] for i, bap in enumerate(baps)]

	start_time = times_baps[0][0]
	end_time = times_baps[-1][0]

	no_zeros = [pair for pair in times_baps if pair[1] < -0.00000001 or pair[1] > 0.00000001]
	bap_timepoints = [start_time]+[pair[0] for pair in no_zeros]+[end_time]

	baps =[pair[1] for pair in no_zeros]
	baps = [baps[0]]+baps+[baps[-1]]

	f = interp.interp1d(bap_timepoints, baps, kind='linear')
	x = np.linspace(start_time, end_time, baps_len)
	return [[v] for v in f(x)]

def preemphasis(x, coef=0.97):
	# is preemphasis only useful for asr or does it help for tts as well??
    b = np.array([1., -coef], x.dtype)
    a = np.array([1.], x.dtype)
    return signal.lfilter(b, a, x)

def inv_preemphasis(x, coef=0.97):
    b = np.array([1.], x.dtype)
    a = np.array([1., -coef], x.dtype)
    return signal.lfilter(b, a, x)

def _delta(x, window):
    return np.correlate(x, window, mode="same")

def _apply_delta_window(x, window):
    T, D = x.shape
    y = np.zeros_like(x)
    for d in range(D):
        y[:, d] = _delta(x[:, d], window)
    return y

def delta_features(x):
    windows = [(0, 0, np.array([1.0])), (1, 1, np.array([-0.5, 0.0, 0.5])), (1, 1, np.array([1.0, -2.0, 1.0]))]
    T, D = x.shape
    assert len(windows) > 0
    combined_features = np.empty((T, D * len(windows)), dtype=x.dtype)
    for idx, (_, _, window) in enumerate(windows):
        combined_features[:, D * idx:D * idx +
                          D] = _apply_delta_window(x, window)
    return combined_features

def extract_features(title, wav_path, bap_mgc_path, times_path, f0_path, delta):

	try:

		fs = 48000

		wav_stream = wavfile.read(os.path.join(wav_path, title + '.wav'))
		wav_stream = np.array(wav_stream[1], dtype=float)

		x = np.array(wav_stream, dtype=np.float64)
		_f0, ts = pw.dio(x, fs)    # raw pitch extractor

		f0 = pw.stonemask(x, _f0, ts, fs)  # pitch refinement
		sp = pw.cheaptrick(x, f0, ts, fs)  # extract smoothed spectrogram
		ap = pw.d4c(x, f0, ts, fs)         # extract aperiodicity

		mgc = pysptk.sp2mc(sp, order=60, alpha=pysptk.util.mcepalpha(fs))
		bap = pysptk.sp2mc(ap, order=5, alpha=pysptk.util.mcepalpha(fs))
		bap = np.concatenate([interpolate_bap(col.flatten()) for col in np.split(bap, len(bap[0]), axis=1)], axis=1)

		vuv = np.array([[0] if i == 0 else [1] for i in f0], dtype=np.float64)

		lf0 = interpolate_f0(f0)

		if delta==True:
			lf0 = delta_features(lf0)
			bap = delta_features(bap)
			mgc = delta_features(mgc)

		output_vector = np.concatenate((vuv, bap, mgc), axis=1)

		with open(os.path.join(f0_path, title + ".lf0"), "w") as h:
			json.dump(lf0.tolist(), h)

		with open(os.path.join(bap_mgc_path, title + ".bap_mgc"), "w") as f:
			json.dump(output_vector.tolist(), f)

		with open(os.path.join(times_path, title + ".times"), "w") as g:
			json.dump(ts.tolist(), g)

	except:
		print(title, 'failed!')


	
if __name__ == '__main__':


	wav_path = sys.argv[1]
	bap_mgc_path = sys.argv[2]
	times_path = sys.argv[3]
	f0_path = sys.argv[4]

	#wav_path = '../build/01_resampled_wav'
	#bap_mgc_path = '../build/02_output_features'
	#times_path = '../build/03_extraction_times'
	#lf0_path = '../build/04_lf0'

	titles = load_titles(wav_path, '.wav')

	# if delta==True use delta features as well
	delta = True

	p = Pool()
	p.starmap(extract_features, [(title, wav_path, bap_mgc_path, times_path, f0_path, delta) for title in titles])

