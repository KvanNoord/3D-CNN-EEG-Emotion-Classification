function [training_data, test_data, training_labels, test_labels] = split_sets_5_fold(E_images, labels)
% Author: Kris van Noord
% Eindhoven University of Technology
% 3D-CNN for EEG emotion classification
% Openly available framework
% Scripts verified on Matlab R2019b


% Set trials in random order
sequence = randperm(40);
E_images = E_images(sequence,:,:,:,:);
labels = labels(sequence,:);

% Calculate number of trials
number_of_trials = size(E_images, 1);

data = zeros(size(E_images,1)*size(E_images,2), size(E_images, 3), size(E_images, 4), size(E_images, 5));
for i = 1:number_of_trials
    data((i-1)*size(E_images,2)+1:i*size(E_images,2),:,:,:) = E_images(i,:,:,:,:);
    labels_subject((i-1)*size(E_images,2)+1:i*size(E_images,2),:) = labels(i,:).* ones(size(E_images,2), 4);
end
data = permute(data, [2 3 4 1]);


number_of_data_points = number_of_trials * size(E_images,2);

% Split in 5 sets for 5-fold classification
test_data(:,:,:,:,1) = data(:,:,:,1:0.2*number_of_data_points);
training_data(:,:,:,:,1) = data(:,:,:,0.2*number_of_data_points+1:end);
test_labels(:,:,1) = labels_subject(1:0.2*number_of_data_points,:);
training_labels(:,:,1) = labels_subject(0.2*number_of_data_points+1:end,:);

test_data(:,:,:,:,2) = data(:,:,:,0.2*number_of_data_points+1:0.4*number_of_data_points);
training_data(:,:,:,:,2) = data(:,:,:,[1:0.2*number_of_data_points, 0.4*number_of_data_points+1:number_of_data_points]);
test_labels(:,:,2) = labels_subject(0.2*number_of_data_points+1:0.4*number_of_data_points,:);
training_labels(:,:,2) = labels_subject([1:0.2*number_of_data_points, 0.4*number_of_data_points+1:number_of_data_points],:);

test_data(:,:,:,:,3) = data(:,:,:,0.4*number_of_data_points+1:0.6*number_of_data_points);
training_data(:,:,:,:,3) = data(:,:,:,[1:0.4*number_of_data_points, 0.6*number_of_data_points+1:number_of_data_points]);
test_labels(:,:,3) = labels_subject(0.4*number_of_data_points+1:0.6*number_of_data_points,:);
training_labels(:,:,3) = labels_subject([1:0.4*number_of_data_points, 0.6*number_of_data_points+1:number_of_data_points],:);

test_data(:,:,:,:,4) = data(:,:,:,0.6*number_of_data_points+1:0.8*number_of_data_points);
training_data(:,:,:,:,4) = data(:,:,:,[1:0.6*number_of_data_points, 0.8*number_of_data_points+1:number_of_data_points]);
test_labels(:,:,4) = labels_subject(0.6*number_of_data_points+1:0.8*number_of_data_points,:);
training_labels(:,:,4) = labels_subject([1:0.6*number_of_data_points, 0.8*number_of_data_points+1:number_of_data_points],:);

test_data(:,:,:,:,5) = data(:,:,:,0.8*number_of_data_points+1:number_of_data_points);
training_data(:,:,:,:,5) = data(:,:,:,1:0.8*number_of_data_points);
test_labels(:,:,5) = labels_subject(0.8*number_of_data_points+1:number_of_data_points,:);
training_labels(:,:,5) = labels_subject(1:0.8*number_of_data_points,:);
end