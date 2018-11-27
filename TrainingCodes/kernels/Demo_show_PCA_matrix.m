% run Demo_Get_PCA_matrix.m first

ksize = 15;

load PCA_P.mat;

% show the PCA basis

for i = 1:size(P,1)
    
    kernel = reshape(P(i,:),ksize,ksize);
    
    subplot 121
    imagesc(kernel);
    title([int2str(i)])
    axis square;
    
    subplot 122
    surf(kernel);
    title([int2str(i)])
    view(45,55)
    xlim([1 15]);
    ylim([1 15]);
    axis square;
    pause(2)
end

