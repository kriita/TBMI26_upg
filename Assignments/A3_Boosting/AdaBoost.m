%% Hyper-parameters

% Number of randomized Haar-features
nbrHaarFeatures = 25;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 500;
% Number of weak classifiers
nbrWeakClassifiers = 30;

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

%% Create image sets (do not modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError
d = ones(1,nbrWeakClassifiers)/nbrWeakClassifiers;
T = nbrWeakClassifiers; 

for t = 1:T
    err = inf;
    
    for k = 1:size(xTrain,1)
        thresh = xTrain(k,:) + 1e-5; 
        
        for i = thresh
            P = 1; % i think?
            C = WeakClassifier(thresh, P, xTrain); % kolla hur många feats (thresh)
            E = WeakClassifierError(C, d, yTrain);
            if E > 0.5
                P = -1;
                E = 1-E;
            end
            
            if E < err
                err = E;
            end
        end
        

    end
end
%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.



%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.



%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.



%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.


