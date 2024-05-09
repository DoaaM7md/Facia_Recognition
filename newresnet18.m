%resnet 18 for feature extraction 26/3/2020
%acuracy   90.39  data set of 10 printer
%acuracy 90.51 with new imagedatastore of datasets of 10 printer


clc
close all
clear
%images=testreaddata();
%D:\Doaa_PHD\PHD2020\DataSets\ck\CK48.......D:\Doaa_PHD\PHD2020\DataSets\jaffedbase - Copy
images=imageDatastore('D:\Doaa_PHD\PHD2020\2nd paper\code\custom cnn48\KDEF','IncludeSubfolders',true,'LabelSource', 'foldernames')
images.ReadFcn = @(filename)readAndPreprocessImage(filename);
[imdsTrain,imdsTest] = splitEachLabel(images,0.7,'randomized');
%Display some sample images.
numImagesTrain = numel(imdsTrain.Labels)
idx = randperm(numImagesTrain,16);

for i = 1:16
    I{i} = readimage(imdsTrain,idx(i));
end

% figure
% imshow(imtile(I))

net =resnet18;
% inputSize = net.Layers(1).InputSize;
% augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
% augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

layer = 'pool5';
%if ~exist('saveTrainingFeaturesnewalexnet.mat', 'file')
featuresTrain = activations(net,imdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,imdsTest,layer,'OutputAs','rows');
%Extract the class labels from the training and test data.
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;
%mdl = fitcecoc(featuresTrain,YTrain);
 %svm 
   %mdl = fitcecoc(featuresTrain,YTrain);
    %ensembel
    %templ = templateTree('Reproducible',true);
    %mdl = fitcensemble(featuresTrain, YTrain,'Method','Bag','NumLearningCycles',100,'Learners',templ); 
    %Naive Base 
    %mdl = fitcnb(featuresTrain, YTrain);
    %KNN
    mdl = fitcknn(featuresTrain, YTrain);
YPred = predict(mdl,featuresTest);
idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    
    imshow(I)
    title(label)
end
accuracy = mean(YPred == YTest)
function Iout = readAndPreprocessImage(filename)
      Iout = imread(filename);
       
    if ismatrix(Iout)
            Iout = cat(3,Iout,Iout,Iout);
        end
        % Resize the image as required for the CNN.6
        Iout = imresize(Iout, [224 224]);
    end