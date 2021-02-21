% Author: Kris van Noord
% Eindhoven University of Technology
% 3D-CNN for EEG emotion classification
% Openly available framework
% Scripts verified on Matlab R2019b


%% Clear workspace
clear all
close all


%% Read in data
% Function input parameters
window_length = 128;                      % window length of sliding window approach (in samples)
step_size = 128;                          % stride or step size of sliding window approach (in samples)
normalization = "Self";                   % Normalization type: No, Self or PreTrial
normalization_operation = "divide" ;      % Normalization operater: divide or subtract
z_score_normalization = "Y";              % Apply Z-Score normalization on final feature vector? (Y/N)  

% The following data is loaded:
% - Differential Entropy of full dataset as [subject, trial, segment, channel, frequency_band] 
% - Labels as [subject, trial, emotion]

% Load Differential Entropy
[DE, labels] = load_data(window_length, step_size, normalization, normalization_operation, z_score_normalization);

% Change labels to classification 0 / 1; 
labels = double(labels > 5);

%% Make EEG images with the spatial information of electrodes
% The data is converted to [subject, trial, segment, 9, 9, frequency_band]
DE = make_eeg_image(DE);

%% Interpolate to 'steps x steps' grid (optional)
% The data is converted to [subject, trial, segment, steps, steps, frequency_band]
steps = 20;
DE = interpolate_grid(DE, steps);

%%
figure()
subplot(2,2,1)
imagesc(reshape(DE(14,32,5,:,:,1), 20, 20))
colorbar
axis off
title("\theta")
caxis([-1, 1])
subplot(2,2,2)
imagesc(reshape(DE(14,32,5,:,:,2), 20, 20))
colorbar
axis off
title("\alpha")
caxis([-1, 1])
subplot(2,2,3)
imagesc(reshape(DE(14,32,5,:,:,3), 20, 20))
colorbar
title("\beta")
axis off
caxis([-1, 1])
subplot(2,2,4)
imagesc(reshape(DE(14,32,5,:,:,4), 20, 20))
colorbar
title("\gamma")
axis off
caxis([-1, 1])

%% Make training / test sets and train CNN per subject

% Loop over subjects
for subject_no = 1:32
    DE_subject = reshape(DE(subject_no,:,:,:,:,:), size(DE, [2 3 4 5 6]));
    labels_subject = reshape(labels(subject_no, :, :), 40, 4);

    % Use 5-fold cross validation
    [training_data, test_data, training_labels, test_labels] = split_sets_5_fold(DE_subject, labels_subject);

    for fold_no = 1:5
        % TrainingOptions
        label_number = 1; % choose 1 for valence, 2 for arousal, 3 for dominance and 4 for liking
        miniBatchSize  = 64;
        validationFrequency = floor(numel(training_labels(:,1))/(1*miniBatchSize));
        options = trainingOptions('adam', ...
            'MiniBatchSize',miniBatchSize, ...
            'MaxEpochs',50, ...
            'InitialLearnRate',1e-4, ...
            'LearnRateSchedule','none', ...
            'Shuffle','every-epoch', ...
            'ValidationData',{test_data(:,:,:,:,fold_no),categorical(test_labels(:,label_number, fold_no), [0,1])}, ...
            'ValidationFrequency',validationFrequency, ...
            'VerboseFrequency', 10000, ...
            'ExecutionEnvironment','gpu', ...
            'Plots','training-progress', ...
            'OutputFcn',@(info)stopIfAccuracyNotImproving(info,10), ...
            'L2Regularization', 0.001);


        % Make model (Yang, 2018)
        layers = [
            imageInputLayer([size(training_data, 1) size(training_data, 2) size(training_data, 3)])
            convolution2dLayer(4,64,'Padding','Same','WeightsInitializer', 'he')
            dropoutLayer(0.5)
            reluLayer
            convolution2dLayer(4,128,'Padding','Same','WeightsInitializer', 'he')
            dropoutLayer(0.5)
            reluLayer
            convolution2dLayer(4,256,'Padding','Same','WeightsInitializer', 'he')            
            dropoutLayer(0.5)
            reluLayer
            convolution2dLayer(1,64,'Padding','Same','WeightsInitializer', 'he')             
            dropoutLayer(0.5)
            reluLayer
            fullyConnectedLayer(1024,'WeightsInitializer', 'he')
            dropoutLayer(0.5)
            reluLayer
            fullyConnectedLayer(2,'WeightsInitializer', 'he')
            softmaxLayer
            classificationLayer];

        [yang2018net, training_info(subject_no,fold_no)] = trainNetwork(training_data(:,:,:,:,fold_no),categorical(training_labels(:,label_number, fold_no), [0,1]),layers,options);

        accuracy(subject_no,fold_no) = max(training_info(subject_no,fold_no).ValidationAccuracy(2:end));

    end
end

