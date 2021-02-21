function interpolated_matrix = interpolate_grid(matrix, steps)
% Author: Kris van Noord
% Eindhoven University of Technology
% 3D-CNN for EEG emotion classification
% Openly available framework
% Scripts verified on Matlab R2019b



% input size should be [subject, trial, segment, width, height, time/frequency]
interpolated_matrix = zeros(size(matrix,1),size(matrix,2),size(matrix, 3), steps,steps,size(matrix, 6));
[xq, yq] = meshgrid(linspace(1,9,steps));
for i = 1:size(matrix, 1)
    for j = 1:size(matrix, 2)
        for n = 1:size(matrix, 3)
            for k = 1:size(matrix, 6)
                image = reshape(matrix(i,j,n,:,:,k), 9, 9);
                [y,x] = find(image ~= 0, 32);
                z = reshape(image(image ~= 0), size(x,1), 1);
                interpolated_matrix(i,j,n,:,:,k) = griddata(x,y,z,xq,yq);
            end
        end
    end
end
interpolated_matrix(isnan(interpolated_matrix)) = 0;
end
