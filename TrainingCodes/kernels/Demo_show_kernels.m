
% load SRMDNFx2.mat;

%% show the Gaussian kernels
%  ksize - e.g., 15, kernel size
%  theta - [0,  pi], rotation angle range
%  l1    - [0.1,10], scaling of eigenvalues
%  l2    - [0.1,l1], scaling of eigenvalues

ksize = 15;


for i = 1:1000
    
    theta = pi*rand(1);
    l1    = 0.1+9.9*rand(1);
    l2    = 0.1+(l1-0.1)*rand(1);
    
    kernel =  anisotropic_Gaussian(ksize,theta,l1,l2);
    
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
    pause(0.1)
end


% %% show the PCA basis
%
%
% for i = 1:size(net.meta.P,1)
%
%     kernel = reshape(net.meta.P(i,:),15,15);
%
%     subplot 121
%     imagesc(kernel);
%     title([int2str(i),' / ',int2str(size(net.meta.P,1))])
%     axis square;
%
%     subplot 122
%     surf(kernel);
%     title([int2str(i),' / ',int2str(size(net.meta.P,1))])
%     view(45,55)
%     xlim([1 15]);
%     ylim([1 15]);
%     axis square;
%     pause(2)
% end

