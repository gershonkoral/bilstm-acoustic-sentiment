function [data,info] = processData(audio,inputSize,info)
    % Break audio into sequences to length inputSize with overlap
    % inputSize/2
    audio = buffer(audio,inputSize,floor(inputSize/2));
    audio = mat2cell(audio,inputSize,ones(1,size(audio,2))).';
    label = repmat(info.Labels,size(audio,1),1);
    
    data = table(audio,label);
end

