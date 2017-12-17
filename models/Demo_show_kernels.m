
load SRMDNFx2.mat;

%% show the Gaussian kernels


kernels = net.meta.AtrpGaussianKernel;

for i = 1:size(kernels,4)
    
    kernel = kernels(:,:,i);
    
    subplot 121
    imagesc(kernel);
    title([int2str(i),' / ',int2str(size(kernels,4))])
    axis square;
    
    subplot 122
    surf(kernel);
    title([int2str(i),' / ',int2str(size(kernels,4))])
    view(45,55)
    xlim([1 15]);
    ylim([1 15]);
    axis square;
    pause(0.1)
end


%% show the PCA basis


for i = 1:size(net.meta.P,1)
    
    kernel = reshape(net.meta.P(i,:),15,15);
    
    subplot 121
    imagesc(kernel);
    title([int2str(i),' / ',int2str(size(net.meta.P,1))])
    axis square;
    
    subplot 122
    surf(kernel);
    title([int2str(i),' / ',int2str(size(net.meta.P,1))])
    view(45,55)
    xlim([1 15]);
    ylim([1 15]);
    axis square;
    pause(2)
end

