%% This script will get optimal k with cross validation
clear
% close all

run setupSupervisedLab.m

%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 1; % Change this to load new data 

% X - Data samples
% D - Desired output from classifier for each sample
% L - Labels for each sample
[X, D, L] = loadDataSet( dataSetNr );

% You can plot and study dataset 1 to 3 by running:
%  plotCase(X,D)

%% Select a subset of the training samples
% 4 bins: 3 for training (2x training 1 val for cross validation), 1 for
% test

numBins = 5;                    % Number of bins you want to devide your data into
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = false;          % true = select samples at random, false = select the first features

[XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom);

% Note: XBins, DBins, LBins will be cell arrays, to extract a single bin from them use e.g.
% XBin1 = XBins{1};
%
% Or use the combineBins helper function to combine several bins into one matrix (good for cross validataion)
% XBinComb = combineBins(XBins, [1,2,3]);

% Add your own code to setup data for training and test here

kmax = 20;
acc_k = zeros(1, kmax);

for k = 1:kmax
    
    acc = 0; % initialize accuracy
   
    for n = 1:numBins % bin n is the validation
        % validation data
        XTest = XBins{n};
        LTest = LBins{n};
        
        % indices for training data in bin
        train_inds = 1:numBins;
        train_inds = train_inds(train_inds ~= n);
        
        XTrain = combineBins(XBins, train_inds);
        LTrain = combineBins(LBins, train_inds);
        
        % Classify training data
        LPredTrain = kNN(XTrain, k, XTrain, LTrain);

        % Classify test data
        LPredTest  = kNN(XTest , k, XTrain, LTrain);
        
        cM = calcConfusionMatrix(LPredTest, LTest);
        
        acc = acc + calcAccuracy(cM); % total sum of accuracies for all n
        
    end
    acc_k(k) = acc / numBins;
    clear acc
end

% calc optimal k
[accuracy, k] = max(acc_k)

figure
plot(acc_k)
ylabel('Accuracy')
xlabel('k')
title('Accuracy of different k in kNN using Cross validation')

% for ds 1: acc 0.9930, k = 30
% for ds 2: acc 0.9990, k = 1
% for ds 3: acc 0.9990, k = 1
% for ds 4: acc 0.9864, k = 4
