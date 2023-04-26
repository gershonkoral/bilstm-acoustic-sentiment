function [dataOut] = preProcess(dataIn,filterOrder)
%  PREPROCESS Applies relevent preprocessing to the input Audio
% 
% Converts the Stereo Audio to Mono and low pass filters the input to reomove 
% frequencies that an adult ear cannot hear (17KHz and above)
fs = 44.1e3;
cutoff = 17e3;
%https://gearspace.com/board/electronic-music-instruments-and-
%electronic-music-production/1172874-converting-stereo-mono-how-does-work.html
%Stereo to Mono Converstion
dataOut = (dataIn(:,1) + dataIn(:,2))/2;
%Remove frequency components above 17Khz
filterVal = designfilt('lowpassiir','FilterOrder',filterOrder, ...
    'HalfPowerFrequency',cutoff,'SampleRate',fs );
dataOut = filtfilt(filterVal,dataOut);
end