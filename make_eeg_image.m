function eeg_image = make_eeg_image(eeg_data)
% Author: Kris van Noord
% Eindhoven University of Technology
% 3D-CNN for EEG emotion classification
% Openly available framework
% Scripts verified on Matlab R2019b


% input size should be [subject, trial, segment, CHANNEL, time/frequency]
matrix_size = size(eeg_data);
matrix = zeros(matrix_size(1), matrix_size(2), matrix_size(3), 9, 9, 4);
for subject = 1:matrix_size(1)
    for trial = 1:matrix_size(2)
        for segment = 1:matrix_size(3)
            matrix(subject, trial, segment, 1, 4, :) = eeg_data(subject, trial, segment, 1, :);
            matrix(subject, trial, segment, 2, 4, :) = eeg_data(subject, trial, segment, 2, :);
            matrix(subject, trial, segment, 3, 3, :) = eeg_data(subject, trial, segment, 3, :);
            matrix(subject, trial, segment, 3, 1, :) = eeg_data(subject, trial, segment, 4, :);
            matrix(subject, trial, segment, 4, 4, :) = eeg_data(subject, trial, segment, 5, :);
            matrix(subject, trial, segment, 4, 2, :) = eeg_data(subject, trial, segment, 6, :);
            matrix(subject, trial, segment, 5, 3, :) = eeg_data(subject, trial, segment, 7, :);
            matrix(subject, trial, segment, 5, 1, :) = eeg_data(subject, trial, segment, 8, :);
            matrix(subject, trial, segment, 6, 2, :) = eeg_data(subject, trial, segment, 9, :);
            matrix(subject, trial, segment, 6, 4, :) = eeg_data(subject, trial, segment, 10, :);
            matrix(subject, trial, segment, 7, 3, :) = eeg_data(subject, trial, segment, 11, :);
            matrix(subject, trial, segment, 7, 1, :) = eeg_data(subject, trial, segment, 12, :);
            matrix(subject, trial, segment, 8, 4, :) = eeg_data(subject, trial, segment, 13, :);
            matrix(subject, trial, segment, 9, 4, :) = eeg_data(subject, trial, segment, 14, :);
            matrix(subject, trial, segment, 9, 5, :) = eeg_data(subject, trial, segment, 15, :);
            matrix(subject, trial, segment, 7, 5, :) = eeg_data(subject, trial, segment, 16, :);
            matrix(subject, trial, segment, 1, 6, :) = eeg_data(subject, trial, segment, 17, :);
            matrix(subject, trial, segment, 2, 6, :) = eeg_data(subject, trial, segment, 18, :);
            matrix(subject, trial, segment, 3, 5, :) = eeg_data(subject, trial, segment, 19, :);
            matrix(subject, trial, segment, 3, 7, :) = eeg_data(subject, trial, segment, 20, :);
            matrix(subject, trial, segment, 3, 9, :) = eeg_data(subject, trial, segment, 21, :);
            matrix(subject, trial, segment, 4, 8, :) = eeg_data(subject, trial, segment, 22, :);
            matrix(subject, trial, segment, 4, 6, :) = eeg_data(subject, trial, segment, 23, :);
            matrix(subject, trial, segment, 5, 5, :) = eeg_data(subject, trial, segment, 24, :);
            matrix(subject, trial, segment, 5, 7, :) = eeg_data(subject, trial, segment, 25, :);
            matrix(subject, trial, segment, 5, 9, :) = eeg_data(subject, trial, segment, 26, :);
            matrix(subject, trial, segment, 6, 8, :) = eeg_data(subject, trial, segment, 27, :);
            matrix(subject, trial, segment, 6, 6, :) = eeg_data(subject, trial, segment, 28, :);
            matrix(subject, trial, segment, 7, 7, :) = eeg_data(subject, trial, segment, 29, :);
            matrix(subject, trial, segment, 7, 9, :) = eeg_data(subject, trial, segment, 30, :);    
            matrix(subject, trial, segment, 8, 6, :) = eeg_data(subject, trial, segment, 31, :);
            matrix(subject, trial, segment, 9, 6, :) = eeg_data(subject, trial, segment, 32, :);
        end
    end
end

eeg_image = matrix;

end
