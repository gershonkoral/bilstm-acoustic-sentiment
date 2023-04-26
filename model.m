clc;
clear;
close all;

% folderName = "../medium"
folderName = "../mono";
% folderName = "../small"
% folderName = "../clippedSorted"
normalizeMethod = "range";
rangeValues = [-1,1];

folder = fullfile(matlabroot,'toolbox','audio','samples');
ads = audioDatastore(folderName,"IncludeSubfolders",true, "LabelSource","foldernames");

[filenames,~] = cellfun(@(x)fileparts(x),ads.Files,'UniformOutput',false);
filenames = (string(filenames));
% wordSplit = split(filenames,"\");
wordSplit = split(filenames,"/");
% segmentation = wordSplit(:,end-1);
segmentation = wordSplit(:,end);


ads.Labels = categorical(segmentation);

%% Processing Data

trainRatio = 0.6;
validationRatio = 0.25;
testRatio = 1-trainRatio-validationRatio;
[train,validation,test] = splitEachLabel(ads,trainRatio,validationRatio,testRatio);

tallTrain      = tall(train);
tallValidation = tall(validation);
tallTest = tall(test);

filterOrder = 12;
%Gets the actual features from the underlying datastore object 

trainFiltered       = cellfun(@(x) preProcess(x,filterOrder),tallTrain,"UniformOutput",false);
validationFiltered =cellfun(@(x) preProcess(x,filterOrder),tallValidation,"UniformOutput",false);
testFiltered =cellfun(@(x) preProcess(x,filterOrder),tallTest,"UniformOutput",false);




%Matlab feature extraction occurs here
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

%Using the power spectrum of the mel-spectrum with 32 bands 
% afe = audioFeatureExtractor( ...
%     'Window',       win, ...
%     'OverlapLength',overlapLength, ...
%     'SampleRate',   fs, ...
%     'melSpectrum',true);
%Defaults to 13 MFCC's 
afe = audioFeatureExtractor( ...
    'Window',       win, ...
    'OverlapLength',overlapLength, ...
    'SampleRate',   fs, ...
    'mfcc',true,...
    'mfccDelta',true, ...
    'mfccDeltaDelta',true);

%Extract features
trainFeatures       = cellfun(@(x)extract(afe,x),trainFiltered,"UniformOutput",false);
validationFeatures =cellfun(@(x)extract(afe,x),validationFiltered,"UniformOutput",false);
testFeatures =cellfun(@(x)extract(afe,x),testFiltered,"UniformOutput",false);

%Normalise Features
trainFeaturesNormed       = cellfun(@(x)normalize(x,normalizeMethod,rangeValues),trainFeatures ,"UniformOutput",false);
validationFeaturesNormed= cellfun(@(x)normalize(x,normalizeMethod,rangeValues),validationFeatures ,"UniformOutput",false);
testFeaturesNormed= cellfun(@(x)normalize(x,normalizeMethod,rangeValues),testFeatures ,"UniformOutput",false);
YTrain = train.Labels;
YValidation = validation.Labels;
YTest = test.Labels;

XTrainData = cellfun(@transpose,trainFeaturesNormed ,'UniformOutput',false);
XTrainData = gather(XTrainData);
XValidationData = cellfun(@transpose,validationFeaturesNormed ,'UniformOutput',false);
XTestData = cellfun(@transpose,testFeaturesNormed,'UniformOutput',false);

XTestData = gather(XTestData);
XValidationData = gather(XValidationData);





summary(YTrain)
summary(YValidation)
summary(YTest)

%% Python feature extraction

path = '/Users/gershikoral/Desktop/labproject/featurescsv';
d = dir(fullfile(path,'*.csv'));
n = length(d);        
pydata = cell(1,n);     
for i=1:n
    pydata{i}=csvread(fullfile(path,d(i).name));
end

pydata_transpose = pydata';
pydata_inside = cellfun(@transpose,pydata_transpose,'UniformOutput',false);

%% Segmenting to the shortest 
% % numObservations = numel(XTrainData);
% for i=1:numObservations
% %     sequence = XTrainData{i};
%     sequence = XValidationData{i};
%     sequenceLengths(i) = size(sequence,2);
% end

% sequenceLengths = sort(sequenceLengths);
% minSequence = sequenceLengths(1);

 
% This only looks at the first 1000 feature points in the data yet we
% ultimately should be using the entire feature set 
%Ensures that all sequences are the same length by only ignoring the final 
% the fearutes outside of the minimum sequence length 
%Alternatively could sort and then use cleverly chosen mini-Batch size to
%prevent padding
% XValidationData  = cellfun(@(x) x(:,1:minSequence), XValidationData ,'uni',false);
% XTrainData = cellfun(@(x) x(:,1:minSequence), XTrainData,'uni',false);
% XTestData = cellfun(@(x) x(:,1:minSequence), XTestData,'uni',false)


%     inputSize = size(XTrainData{1,1},1);
%     numHiddenUnits = 100;
%     numClasses = 2;
%     maxEpochs = 100;
%     miniBatchSize = 20;
%     initialLearnRate = 0.00001;
% %     
%     %Understanding the basic layers of an LSTM
% %     %https://towardsdatascience.com/reading-between-the-layers-lstm-network-7956ad192e58
%    layers = [ ...
%     sequenceInputLayer(inputSize)
%     bilstmLayer(numHiddenUnits,"OutputMode","last")
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer]
%    
%    XValidationData = gather(XValidationData);
%    
%    options = trainingOptions('adam', ...
%         'GradientThreshold',1,...
%         'ExecutionEnvironment','cpu', ...
%         'MaxEpochs',maxEpochs, ...
%         'MiniBatchSize',miniBatchSize, ...
%         'InitialLearnRate',initialLearnRate,...
%         'SequenceLength','shortest', ... % Crops rather than padds
%         'ValidationData',{XValidationData,YValidation}, ...
%         'ValidationFrequency',50,... % The number of iterations between evaluations of validation metrics
%         'Shuffle','every-epoch', ... % Shuffle the training data before each training epoch, and shuffle the validation data before each network validation. To avoid discarding the same data every epoch, set the 'Shuffle' value to 'every-epoch'.
%         'Verbose',1, ... %produces a summary of training https://www.mathworks.com/help/deeplearning/ref/trainingoptions.html#bu80qkw-3_head
%         'VerboseFrequency',50,... %number of iterations between printing summary
%         'Plots','training-progress')
%      
%     net = trainNetwork(XTrainData,YTrain,layers,options);

%% No shuffling and Mini Batch Sizes

sequenceLengths = cellfun(@(X) size(X,2), XTrainData);
[sequenceLengthsSorted,idx] = sort(sequenceLengths);
XTrainData = XTrainData(idx);
YTrain = YTrain(idx);

sequenceLengths = cellfun(@(X) size(X,2), XValidationData);
[sequenceLengthsSorted,idx] = sort(sequenceLengths);
XValidationData = XValidationData(idx);
YValidation = YValidation(idx);


figure
bar(sequenceLengthsSorted)
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")

inputSize = size(XTrainData{1,1},1);
    numHiddenUnits = 100;
    numClasses = 2;
    maxEpochs = 100;
    miniBatchSize = 50;
    initialLearnRate = 0.001;
%     
    %Understanding the basic layers of an LSTM
%     %https://towardsdatascience.com/reading-between-the-layers-lstm-network-7956ad192e58
   layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,"OutputMode","last")
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]
   
   XValidationData = gather(XValidationData);
   
   options = trainingOptions('adam', ...
        'GradientThreshold',1,...
        'ExecutionEnvironment','cpu', ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ...
        'InitialLearnRate',initialLearnRate,...
        'SequenceLength','shortest', ... % Crops rather than padds
        'ValidationData',{XValidationData,YValidation}, ...
        'ValidationFrequency',50,... % The number of iterations between evaluations of validation metrics
        'Shuffle','Never', ... % Shuffle the training data before each training epoch, and shuffle the validation data before each network validation. To avoid discarding the same data every epoch, set the 'Shuffle' value to 'every-epoch'.
        'Verbose',1, ... %produces a summary of training https://www.mathworks.com/help/deeplearning/ref/trainingoptions.html#bu80qkw-3_head
        'VerboseFrequency',50,... %number of iterations between printing summary
        'Plots','training-progress')
     
    net = trainNetwork(XTrainData,YTrain,layers,options);
%% Testing the model

XTestData = gather(XTestData);
YPred = classify(net,XTestData, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','shortest');

acc = sum(YPred == YTest)./numel(YTest)

figure('Name','Confusion Chart');
confusionchart(YTest,YPred)
