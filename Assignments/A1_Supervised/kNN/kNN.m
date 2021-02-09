function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)

classes = unique(LTrain);
NClasses = length(classes);

% Add your own code here
LPred  = zeros(size(X,1),1);

% for each sample i in X, compare distance to all XTrain samples, get the k
% closest ones, get their labels, set the LPred for this sample as the
% majority of the labels. If k is even, check the total distance for these
% close ones and see which one is closest.

for i = 1:size(X,1)
    dist = pdist2( X(i,:), XTrain); %vec with distances from sample i to all training samples
    [~, index] = sort(dist);
    NN_indexes = index(1:k); % indexes of k nearest neighbours
    NN_classes = LTrain(NN_indexes);
    [most_common, freq] = mode(NN_classes);
    % check if there is a tie
    
    warning('off','all')
    [sec_most, sec_freq] = mode(NN_classes(NN_classes ~= most_common));
    warning('on','all')

    if freq == sec_freq % if a tie
        
        % sorted list of classes in distance order
        possible_choices = NN_classes(NN_classes == most_common | NN_classes == sec_most); 
        most_common = possible_choices(1); % choose the nearest one
        
    end
     
     LPred(i)= most_common;
end
    
end

