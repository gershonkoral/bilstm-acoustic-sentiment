clc;
clear;
close all;

folderName = "/Users/gershikoral/Desktop/labproject/mono";

normalizeMethod = "range";
rangeValues = [-1,1];
standardizationMethod = "zscore";

folder = fullfile(matlabroot,'toolbox','audio','samples');
ads = audioDatastore(folderName,"IncludeSubfolders",true, "LabelSource","foldernames");

[filenames,~] = cellfun(@(x)fileparts(x),ads.Files,'UniformOutput',false);
filenames = (string(filenames));
wordSplit = split(filenames,"/");
segmentation = wordSplit(:,end);


ads.Labels = categorical(segmentation);

clear folderName filenames wordSplit segmentation

trainRatio = 0.6;
validationRatio = 0.25;
testRatio = 1-trainRatio-validationRatio;
shuf = shuffle(ads);
[train,validation,test] = splitEachLabel(shuf,trainRatio,validationRatio,testRatio,'randomized');

tallTrain      = readall(train);
tallValidation = readall(validation);
tallTest = readall(test);

clear trainRatio validationRatio testRatio 

%% Matlab feature extraction occurs here
[~,adsInfo] = read(ads);
fs = adsInfo.SampleRate;

%Produces a frame size of 30ms
%https://www.mathworks.com/help/audio/ug/sequential-feature-selection-for-audio-features.html#mw_rtc_SequentialFeatureSelectionForAudioFeaturesExample_FDEE1A23
%Code above specifies that this sets the overlap and window size as
%required
%t_n = n*T ( the location of the sample = time step * Period)
win = hamming(round(0.03*fs),"periodic"); 
%Produces an overlap length of 15ms --> 50% of the original
overlapLength = round(0.015*fs);


afe = audioFeatureExtractor( ...
    'Window',       win, ...
    'OverlapLength',overlapLength, ...
    'SampleRate',   fs, ...
     "SpectralDescriptorInput","melSpectrum", ...
    'mfccDeltaDelta', true);

setExtractorParams(afe,"melSpectrum","NumBands",26,"FrequencyRange",[20 17e3]);

%Extract features
trainFeatures       = cellfun(@(x)extract(afe,x),tallTrain,"UniformOutput",false);
validationFeatures =cellfun(@(x)extract(afe,x),tallValidation,"UniformOutput",false);
testFeatures =cellfun(@(x)extract(afe,x),tallTest,"UniformOutput",false);


%Standardization
trainFeaturesStandardise        = cellfun(@(x)normalize(x, standardizationMethod),trainFeatures ,"UniformOutput",false);
%Normalise Features
trainFeaturesNormed       = cellfun(@(x)normalize(x,normalizeMethod,rangeValues),trainFeaturesStandardise  ,"UniformOutput",false);


validationFeaturesStandardise = cellfun(@(x)normalize(x,standardizationMethod),validationFeatures ,"UniformOutput",false);
testFeaturesStandardize= cellfun(@(x)normalize(x,standardizationMethod),testFeatures ,"UniformOutput",false);


validationFeaturesNormed= cellfun(@(x)normalize(x,normalizeMethod,rangeValues),validationFeaturesStandardise ,"UniformOutput",false);
testFeaturesNormed= cellfun(@(x)normalize(x,normalizeMethod,rangeValues),testFeaturesStandardize ,"UniformOutput",false);
YTrain = train.Labels;
YValidation = validation.Labels;
YTest = test.Labels;

XTrainData = cellfun(@transpose,trainFeaturesNormed ,'UniformOutput',false);
XTrainData = gather(XTrainData);
XValidationData = cellfun(@transpose,validationFeaturesNormed ,'UniformOutput',false);
XTestData = cellfun(@transpose,testFeaturesNormed,'UniformOutput',false);

XTestData = gather(XTestData);
XValidationData = gather(XValidationData);


%% Segmenting to the shortest 
numObservations = numel(XTrainData);
for i=1:numObservations
    sequence = XTrainData{i};
    %sequence = XValidationData{i};
    sequenceLengths(i) = size(sequence,2);
end

sequenceLengths = sort(sequenceLengths);
minSequence = sequenceLengths(1);

XValidationData  = cellfun(@(x) x(:,1:minSequence), XValidationData ,'uni',false);
XTrainData = cellfun(@(x) x(:,1:minSequence), XTrainData,'uni',false);
XTestData = cellfun(@(x) x(:,1:minSequence), XTestData,'uni',false);

%% Lets reg baby

all_data = vertcat(XTrainData, XValidationData);
all_data = vertcat(all_data, XTestData);

all_labels = vertcat(YTrain, YValidation);
all_labels = vertcat(all_labels, YTest);

a = size(all_data{1});
col_av_cell = cell(a(1),1); 

for i=1:a
    col_av_cell{i,1} = cellfun(@(x) x(i,:), all_data, 'UniformOutput',false);
    col_av_cell{i,1} = cell2mat(col_av_cell{i,1});
end

%%
feature1 = cell2mat(col_av_cell(4,1));
labels_ = double(all_labels);
feature1 = horzcat(feature1, labels_);