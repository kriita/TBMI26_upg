function [ cM ] = calcConfusionMatrix( LPred, LTrue )
% CALCCONFUSIONMATRIX returns the confusion matrix of the predicted labels

classes  = unique(LTrue);
NClasses = length(classes);

% Add your own code here
cM = zeros(NClasses);

for i = 1:length(classes)
    row_class = classes(i);
    for j = 1:length(classes)
        col_class = classes(j);
        cM(i,j) = sum((LPred == col_class).*(LTrue == row_class));
    end
end

