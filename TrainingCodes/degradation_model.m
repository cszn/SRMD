function [LR, HR, kernel, sigma] = degradation_model(HR, scale)


% kernel
ksize  = 15;
theta  = pi*rand(1);
l1     = 0.1+9.9*rand(1);
l2     = 0.1+(l1-0.1)*rand(1);
kernel = anisotropic_Gaussian(ksize,theta,l1,l2); % double
kernel = single(kernel);

% noise level
sigma_max = 75;
sigma = single(sigma_max*rand(1)/255); % single

% HR image
HR  = modcrop(HR, scale); % double

% xxxxxxxxxxxxxx degradation model xxxxxxxxxxxxxxxxxxxxxx
% you can change to your own degradation model

blurry_HR  = imfilter(HR,double(kernel),'replicate');
downsampled_HR = imresize(blurry_HR, 1/scale,'bicubic');
downsampled_HR = im2single(im2uint8(downsampled_HR));

noise  = single(randn(size(downsampled_HR),'single')*sigma);
LR = downsampled_HR + noise;

kernel = single(kernel);
% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

end















