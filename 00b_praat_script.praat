form Script arguments
	sentence Title 
	sentence Wav_path 
	sentence F0_path
endform


Read from file: wav_path$ + title$ + ".wav"
selectObject: "Sound " + title$

To Pitch (ac)... 0.005 75 15 no 0.1 0.8 0.01 0.8 0.14 600

Down to PitchTier

selectObject: "PitchTier " + title$
Interpolate quadratically... 4 semitones

sr = 0.005
length = Get end time
npts = round(length/sr)

deleteFile: f0_path$ + title$ + ".f0"

for i from 1 to npts
	f0 = Get value at time... (i-1)*sr
	appendFileLine: f0_path$ + title$ + ".f0", f0
endfor
