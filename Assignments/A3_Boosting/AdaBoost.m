%% Hyper-parameters
clear
close all
clc

% Number of randomized Haar-features
nbrHaarFeatures = 100;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 1000;
% Number of weak classifiers
nbrWeakClassifiers = 100;

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
d = ones(1,size(xTrain,2))/size(xTrain,2);

thresholdopt = zeros(nbrWeakClassifiers, 1);
polopt = zeros(nbrWeakClassifiers, 1);
haaropt = zeros(nbrWeakClassifiers, 1);
alphopt = zeros(nbrWeakClassifiers, 1);

for t = 1:nbrWeakClassifiers
    err = inf;
    
    for k = 1:nbrHaarFeatures
        thr = xTrain(k,:) + 1e-7;
        
        for i = 1:length(thr)
            P = 1;
            thresh = thr(i);
            C = WeakClassifier(thresh, P, xTrain(k,:));
            E = WeakClassifierError(C, d, yTrain);
            if E > 0.5
                P = -P;
                E = 1-E;
            end
            
            if E < err
                err = E;
                alphopt(t) = 1/2 * log((1 - err) / err);
                thresholdopt(t) = thresh;
                polopt(t) = P;
                haaropt(t) = k;
                Copt = P * C;
            end
        end
    end
    
    d = d .* exp(-alphopt(t) * yTrain .* Copt);
    
    d = d/sum(d);
end

%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.

class_Tr = zeros(length(thresholdopt), size(yTrain,2));
class_Te = zeros(length(thresholdopt), size(yTest,2));
for i = 1:size(class_Tr,1)
    class_Tr(i,:) = alphopt(i) * WeakClassifier(thresholdopt(i), polopt(i), xTrain(haaropt(i), :));
    class_Te(i,:) = alphopt(i) * WeakClassifier(thresholdopt(i), polopt(i), xTest(haaropt(i), :));
end

class_Tr = sign(sum(class_Tr, 1));
class_Te = sign(sum(class_Te, 1));

acc_Tr = 1 - mean(abs(class_Tr - yTrain))/2;
acc_Te = 1 - mean(abs(class_Te - yTest))/2;

%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.

acc_Tr_diff = zeros(1,nbrWeakClassifiers);
acc_Te_diff = zeros(1,nbrWeakClassifiers);

for i = 1:length(acc_Tr_diff)
    class_Tr = zeros(length(thresholdopt), size(yTrain,2));
    class_Te = zeros(length(thresholdopt), size(yTest,2));
    for j = 1:i
        class_Tr(j,:) = alphopt(j) * WeakClassifier(thresholdopt(j), polopt(j), xTrain(haaropt(j), :));
        class_Te(j,:) = alphopt(j) * WeakClassifier(thresholdopt(j), polopt(j), xTest(haaropt(j), :));
    end
    class_Tr = sign(sum(class_Tr, 1));
    class_Te = sign(sum(class_Te, 1));
    acc_Tr_diff(i) = 1 - mean(abs(class_Tr - yTrain))/2;
    acc_Te_diff(i) = 1 - mean(abs(class_Te - yTest))/2;
end

figure
plot(1-acc_Tr_diff)
hold on
plot(1-acc_Te_diff)
hold off
xlabel("Number of Classifiers")
ylabel("Accuracy")
legend("Training data", "Test data")

%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.

misclass = class_Te ~= yTest;
misclass_loc = find(misclass);

figure
for k = 1:9
    subplot(3,3,k),imagesc(testImages(:,:,misclass_loc(40*k)));
    axis image;
    axis off;
end


%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.


figure
for k = 1:9
    subplot(3,3,k),imagesc(haarFeatureMasks(:,:,k*10));
    axis image;
    axis off;
end