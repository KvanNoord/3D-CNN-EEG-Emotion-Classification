function diff_entropy = calculate_DE(signals)
% Author: Kris van Noord
% Eindhoven University of Technology
% 3D-CNN for EEG emotion classification
% Openly available framework
% Scripts verified on Matlab R2019b

variances = var(signals,1);
diff_entropy = 0.5 * log(2*pi*exp(1)*variances);
end
