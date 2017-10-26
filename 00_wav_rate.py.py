import subprocess
import os

sr = 48000

titles = []
for fn in os.listdir('wav/'):
	titles.append(fn)

os.makedirs('wav_')
for title in titles:
	subprocess.call(['sox', 'wav/'+title, '-b', '16', 'wav_/'+title, 'rate', str(sr)])