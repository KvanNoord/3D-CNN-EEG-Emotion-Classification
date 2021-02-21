function [x_data, EEG_labels] = load_data(window_length, step_size, normalization, normalization_operation, z_score_normalization)
% Author: Kris van Noord
% Eindhoven University of Technology
% 3D-CNN for EEG emotion classification
% Openly available framework
% Scripts verified on Matlab R2019b


% Load raw data
% Datafile should have size [32, 40, 32, 8064] !!
EEG_data = load('EEG_data.mat');
EEG_data = EEG_data.new_data;

% Load labels
EEG_labels = load('EEG_labels');
EEG_labels = EEG_labels.EEG_labels;

% Calculate filter coefficients for bandpass filter
[b1, a1] = butter(6,[4 8]./64,'bandpass');
[b2, a2] = butter(6,[8 12]./64,'bandpass');
[b3, a3] = butter(6,[12 30]./64,'bandpass');
[b4, a4] = butter(6,[30 63]./64,'bandpass');

% Calculate number of segments with current window_size & step_size
number_of_segments = round(7680/step_size - (window_length / step_size - 1));

% Calculate Differential Entropy for all trials
x_data = zeros(32,40,number_of_segments,32,4);
for subject = 1 : 32
    for trial = 1 : 40
        % Filter the signal into 4 different signals
        S_bp_1 = filter(b1, a1, reshape(EEG_data(subject, trial, :, :), 32, 8064).').';
        S_bp_2 = filter(b2, a2, reshape(EEG_data(subject, trial, :, :), 32, 8064).').';
        S_bp_3 = filter(b3, a3, reshape(EEG_data(subject, trial, :, :), 32, 8064).').';
        S_bp_4 = filter(b4, a4, reshape(EEG_data(subject, trial, :, :), 32, 8064).').';
        
        % Define baseline signals (first 384 samples / 3 seconds)
        baseline_1 = S_bp_1(:, 1:384);
        baseline_2 = S_bp_2(:, 1:384);
        baseline_3 = S_bp_3(:, 1:384);
        baseline_4 = S_bp_4(:, 1:384);
        
        % Calculate DE of every segment in baseline
        DE_baseline_1 = calculate_DE(baseline_1.');
        DE_baseline_2 = calculate_DE(baseline_2.');
        DE_baseline_3 = calculate_DE(baseline_3.');
        DE_baseline_4 = calculate_DE(baseline_4.');
        
        % Calculate average DE in baseline
        baseline_DE(1,:) = mean(DE_baseline_1, 1);
        baseline_DE(2,:) = mean(DE_baseline_2, 1);
        baseline_DE(3,:) = mean(DE_baseline_3, 1);
        baseline_DE(4,:) = mean(DE_baseline_4, 1);
        
        % Calculate DE for all segments and (if desired) normalize
        for i = 1:number_of_segments
            if normalization == "PreTrial" 
                signal_1 = calculate_DE(S_bp_1(:, (i-1)*step_size+385:(i-1)*step_size + window_length + 384).');
                signal_2 = calculate_DE(S_bp_2(:, (i-1)*step_size+385:(i-1)*step_size + window_length + 384).');
                signal_3 = calculate_DE(S_bp_3(:, (i-1)*step_size+385:(i-1)*step_size + window_length + 384).');
                signal_4 = calculate_DE(S_bp_4(:, (i-1)*step_size+385:(i-1)*step_size + window_length + 384).');                
               
                if normalization_operation == "divide"
                    x_data(subject,trial,i,:,1) = signal_1 ./ DE_baseline_1;
                    x_data(subject,trial,i,:,2) = signal_2 ./ DE_baseline_2;
                    x_data(subject,trial,i,:,3) = signal_3 ./ DE_baseline_3;
                    x_data(subject,trial,i,:,4) = signal_4 ./ DE_baseline_4;
                elseif normalization_operation == "subtract"
                    x_data(subject,trial,i,:,1) = signal_1 - DE_baseline_1;
                    x_data(subject,trial,i,:,2) = signal_2 - DE_baseline_2;
                    x_data(subject,trial,i,:,3) = signal_3 - DE_baseline_3;
                    x_data(subject,trial,i,:,4) = signal_4 - DE_baseline_4;           
                else
                    disp("Error: wrong normalization_operation")
                end
            elseif normalization == "Self"
                x_data(subject,trial,i,:,1) = calculate_DE(S_bp_1(:, (i-1)*step_size+385:(i-1)*step_size + window_length + 384).');
                x_data(subject,trial,i,:,2) = calculate_DE(S_bp_2(:, (i-1)*step_size+385:(i-1)*step_size + window_length + 384).');
                x_data(subject,trial,i,:,3) = calculate_DE(S_bp_3(:, (i-1)*step_size+385:(i-1)*step_size + window_length + 384).');
                x_data(subject,trial,i,:,4) = calculate_DE(S_bp_4(:, (i-1)*step_size+385:(i-1)*step_size + window_length + 384).');
                if normalization_operation == "divide"
                    x_data(subject,trial,i,:,:) = x_data(subject,trial,i,:,:) ./ mean(x_data(subject,trial,i,:,:), 5);
                elseif normalization_operation == "subtract"    
                    x_data(subject,trial,i,:,:) = x_data(subject,trial,i,:,:) - mean(x_data(subject,trial,i,:,:), 5);
                else
                    disp("Error: wrong normalization_operation")
                end
                
            elseif normalization == "No"
                x_data(subject,trial,i,:,1) = calculate_DE(S_bp_1(:, (i-1)*step_size+385:(i-1)*step_size + window_length + 384).');
                x_data(subject,trial,i,:,2) = calculate_DE(S_bp_2(:, (i-1)*step_size+385:(i-1)*step_size + window_length + 384).');
                x_data(subject,trial,i,:,3) = calculate_DE(S_bp_3(:, (i-1)*step_size+385:(i-1)*step_size + window_length + 384).');
                x_data(subject,trial,i,:,4) = calculate_DE(S_bp_4(:, (i-1)*step_size+385:(i-1)*step_size + window_length + 384).');
            else
                disp("Error: wrong normalization")
            end    
            % Apply Z-Score normalization (ref: Yang et al, 2018) (OPTION)
            if z_score_normalization == "Y"
                x_data(subject,trial,i,:,1) = (x_data(subject,trial,i,:,1) - mean(x_data(subject,trial,i,:,1))) ./ std(x_data(subject,trial,i,:,1));
                x_data(subject,trial,i,:,2) = (x_data(subject,trial,i,:,2) - mean(x_data(subject,trial,i,:,2))) ./ std(x_data(subject,trial,i,:,2));
                x_data(subject,trial,i,:,3) = (x_data(subject,trial,i,:,3) - mean(x_data(subject,trial,i,:,3))) ./ std(x_data(subject,trial,i,:,3));
                x_data(subject,trial,i,:,4) = (x_data(subject,trial,i,:,4) - mean(x_data(subject,trial,i,:,4))) ./ std(x_data(subject,trial,i,:,4));
            end
        end       
    end
end
end
