clc;
clear;
close all;

folderName = "../mono"; % read .wav files here
normalizeMethod = "range";
rangeValues = [-1,1];

folder = fullfile(matlabroot,'toolbox','audio','samples');
ads = audioDatastore(folderName,"IncludeSubfolders",true, "LabelSource","foldernames");

[filenames,~] = cellfun(@(x)fileparts(x),ads.Files,'UniformOutput',false);
filenames = (string(filenames));
wordSplit = split(filenames,"/");
segmentation = wordSplit(:,end);


labels = categorical(segmentation);

clear ads segmentation wordSplit 

%% Python feature extraction

path = '../featurescsv/False'; % read feature csv files here
d = dir(fullfile(path,'*.csv'));
n = length(d);        
pydata_false = cell(1,n);     
for i=1:n
    pydata_false{i}=csvread(fullfile(path,d(i).name));
end

pydata_transpose_false = pydata_false';
pydata_inside_false = cellfun(@transpose,pydata_transpose_false,'UniformOutput',false);

path_true = '../featurescsv/True';
d = dir(fullfile(path_true,'*.csv'));
n = length(d); 
pydata_true = cell(1,n);     
for i=1:n
    pydata_true{i}=csvread(fullfile(path_true,d(i).name));
end

pydata_transpose_true = pydata_true';
pydata_inside_true = cellfun(@transpose,pydata_transpose_true,'UniformOutput',false);

pydata = vertcat(pydata_inside_false,pydata_inside_true);

% pydata = cellfun(@(x)normalize(x,normalizeMethod,rangeValues),pydata ,"UniformOutput",false);

pydata = cellfun(@(x) x([1 2 3 22:34], :), pydata, 'UniformOutput', false); %time domain

%% Lets pre-process baby
trainRatio = 0.6;
validationRatio = 0.25;
testRatio = 1-trainRatio-validationRatio;
% [train,validation,test] = splitEachLabel(ads,trainRatio,validationRatio,testRatio);

%% 

[trainInd,valInd,testInd] = divideint(height(pydata),trainRatio, validationRatio, testRatio);
[YtrainInd,YvalInd,YtestInd] = divideint(height(labels),trainRatio, validationRatio, testRatio);

XTrainData = pydata(trainInd, :);
XValidationData = pydata(valInd, :);
XTestData = pydata(testInd, :);

YTrain = labels(YtrainInd, :);
YValidation = labels(YvalInd, :);
YTest = labels(YtestInd, :);

clear trainInd valInd testInd YtrainInd YvalInd YtestInd
% 
% %% Stats of training
% 
% % False ZCR train
% zcr_false = 0;
% energy_false = 0;
% for i=1:nnz(YTrain=='False')
%     a = mean(XTrainData{i,1}(1,:));
%     zcr_false = zcr_false + a;
%     b = mean(XTrainData{i,1}(2,:));
%     energy_false = energy_false + b;
% end
% 
% % True ZCR train
% zcr_true = 0;
% energy_true = 0;
% for i=(nnz(YTrain=='False')+1):size(YTrain)
%     a = mean(XTrainData{i,1}(1,:));
%     zcr_true = zcr_true + a;
%     b = mean(XTrainData{i,1}(2,:));
%     energy_true = energy_true + b;
% end
% 
% %% Stats of validation
% 
% % False ZCR train
% zcr_false_val = 0;
% energy_false_val = 0;
% for i=1:nnz(YValidation=='False')
%     a = mean(XValidationData{i,1}(1,:));
%     zcr_false_val = zcr_false_val + a;
%     b = mean(XValidationData{i,1}(2,:));
%     energy_false_val = energy_false_val + b;
% end
% 
% % True ZCR train
% zcr_true_val = 0;
% energy_true_val = 0;
% for i=(nnz(YValidation=='False')+1):size(YValidation)
%     a = mean(XValidationData{i,1}(1,:));
%     zcr_true_val = zcr_true_val + a;
%     b = mean(XValidationData{i,1}(2,:));
%     energy_true_val = energy_true_val + b;
% end

% plot(1:1997, XTrainData{1,1}(1,:)) % ZCR for audio clip 1
%% BiLSTM Network Architecture

sequenceLengths = cellfun(@(X) size(X,2), XTrainData);
[sequenceLengthsSorted,idx] = sort(sequenceLengths);
XTrainData = XTrainData(idx);
YTrain = YTrain(idx);

sequenceLengths = cellfun(@(X) size(X,2), XValidationData);
[sequenceLengthsSorted,idx] = sort(sequenceLengths);
XValidationData = XValidationData(idx);
YValidation = YValidation(idx);

% figure
% bar(sequenceLengthsSorted)
% xlabel("Sequence")
% ylabel("Length")
% title("Sorted Data")

inputSize = size(XTrainData{1,1},1);
numHiddenUnits = 11;
numClasses = 2;
maxEpochs = 100;
miniBatchSize = 25;
initialLearnRate = 0.0144;
drop = 0.2932;
l2reg = 0.0079;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,"OutputMode","last")
    batchNormalizationLayer
    dropoutLayer(drop)
    batchNormalizationLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
   
XValidationData = gather(XValidationData);

options = trainingOptions('adam', ...
    'GradientThreshold',1,...
    'ExecutionEnvironment','cpu', ...
    'L2Regularization',l2reg,...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',initialLearnRate,...
    'SequenceLength','shortest', ... % Crops rather than padds
    'ValidationData',{XValidationData,YValidation}, ...
    'ValidationFrequency',50,... % The number of iterations between evaluations of validation metrics
    'Shuffle','every-epoch', ... % Shuffle the training data before each training epoch, and shuffle the validation data before each network validation. To avoid discarding the same data every epoch, set the 'Shuffle' value to 'every-epoch'.
    'Verbose',1, ... %produces a summary of training https://www.mathworks.com/help/deeplearning/ref/trainingoptions.html#bu80qkw-3_head
    'VerboseFrequency',50,... %number of iterations between printing summary
    'Plots','training-progress');
   
net = trainNetwork(XTrainData,YTrain,layers,options);

%% Testing the model
XTestData = gather(XTestData);
YPred = classify(net,XTestData, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','shortest');

acc = sum(YPred == YTest)./numel(YTest)


figure('Name','Confusion Chart');
cm =confusionchart(YTest,YPred);


TN = cm.NormalizedValues(1,1);
TP= cm.NormalizedValues(2,2);
FP =cm.NormalizedValues(1,2);
FN =cm.NormalizedValues(2,1);
Total = TN +TP+FP+FN;
accuracy =(TP+TN)/Total


TPR = TP/(FN+TP);
TNR = TN/(TN+FP);
FPR = FP/(TN+FP);
Presion = TP/(TP+FP)
Recall = TP/(TP+FN)
f1_pres = 1/Presion;
f1_recall = 1/Recall;
F1 = 2*(1/(f1_pres + f1_recall))
%https://towardsdatascience.com/the-3-most-important-composite-classification-metrics-b1f2d886dc7b
balanced_accuracy = (TPR+ TNR)/2
actual = YTest =='True';
pred = YPred =='True';
[X,Y,T,AUC,OPTROCPT] = perfcurve(double(actual),double(pred),1);
AUC

% plot(X,Y)
num = TP*TN-FP*FN;
denom = (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN);
MCC = num / sqrt(denom)
