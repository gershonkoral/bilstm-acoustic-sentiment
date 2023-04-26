# Laboratory Code

To run non-interactively on VM:

`./matlab -nodisplay -nosplash -nodesktop -r "run('model.m');exit;"`

To batch run feature extraction:
File location --> `cd ../pyAudioAnalysis/pyAudioAnalysis`

`python3 audioAnalysis.py featureExtractionDir -i data/ -mw 1.0 -ms 1.0 -sw 0.050 -ss 0.015`
