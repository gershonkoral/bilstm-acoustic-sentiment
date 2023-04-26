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

% pydata = cellfun(@(x) x([1 2 3 22:34], :), pydata, 'UniformOutput', false); %time domain

%%
a = size(pydata{1});
col_av_cell = cell(a(1),1); 

for i=1:a
    col_av_cell{i,1} = cellfun(@(x) x(i,:), pydata, 'UniformOutput',false);
    col_av_cell{i,1} = cell2mat(col_av_cell{i,1});
end

%%
feature1 = cell2mat(col_av_cell(10,1));
labels_ = double(labels);
feature1 = horzcat(feature1, labels_);

