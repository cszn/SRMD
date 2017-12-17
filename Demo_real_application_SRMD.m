%==========================================================================
% This is the testing code of SRMD (x2, x3, x4) for real image SR.
% For general degradation, the basic setting is:
%   1. there are tree types of kernels, including isotropic Gaussian,
%      anisotropic Gaussian, and estimated kernel k_b for isotropic
%      Gaussian k_d under direct downsampler (x2 and x3 only).
%      It is preferred to estimate the kernel first, or you can sample
%      several kernels to produce multiple results and select the best one.
%   2. the noise level range is [0, 75].
%   3. the downsampler is fixed to bicubic downsampler.
%      For direct downsampler, you can either train a new model with
%      direct downsamper or use the estimated kernel k_b under direct
%      downsampler. The former is preferred.
%   4. there are three models, "SRMDx2.mat" for scale factor 2, "SRMDx3.mat"
%      for scale factor 3, and "SRMDx4.mat" for scale factor 4.
%==========================================================================
% The basic idea of SRMD is to learn a CNN to infer the MAP of general SISR, i.e.,
% solve x^ = arg min_x 1/(2 sigma^2) ||(kx)\downarrow_s - y||^2 + lamda \Phi(x)
% via x^ = CNN(y,k,sigma;\Theta) or HR^ = CNN(LR,kernel,noiselevel;\Theta).
%
% There involves two important factors, i.e., blur kernel (k; kernel) and noise
% level (sigma; nlevel).
%
% For more information, please refer to the following paper.
%    @article{zhang2017learningsrmd,
%    title={Learning a Single Convolutional Super-Resolution Network for Multiple Degradations},
%    author={Kai, Zhang and Wangmeng, Zuo and Lei, Zhang},
%    year={2017},
%    }
%
% If you have any question, please feel free to contact with <Kai Zhang (cskaizhang@gmail.com)>.
%
% This code is for research purpose only.
%
% by Kai Zhang (Nov, 2017)
%==========================================================================

% clear; clc;
format compact;

addpath('utilities');
imageSets    = {'chip','cat','flowers','stars','Set5','Set14','BSD100','Urban100'}; % testing dataset


%%======= ======= ======= degradation parameter settings ======= ======= =======

% For real image 'chip', some examples of degradation setting are given as follows.
% sf = 2; nlevel = 5~10; kerneltype = 1; kernelwidth = 0.8;
% sf = 2; nlevel = 5~10; kerneltype = 3; nk          = 5;
% sf = 3; nlevel = 5~10; kerneltype = 1; kernelwidth = 1.2;
% sf = 3; nlevel = 5~10; kerneltype = 3; nk          = 5;
% sf = 4; nlevel = 5~10; kerneltype = 1; kernelwidth = 1.7;

% For real image 'cat', some examples of degradation setting are given as follows.
% sf = 2; nlevel = 20; kerneltype = 1; kernelwidth = 1.6;
% sf = 2; nlevel = 20; kerneltype = 3; nk          = 12;
% sf = 3; nlevel = 20; kerneltype = 1; kernelwidth = 2.4;
% sf = 3; nlevel = 20; kerneltype = 3; nk          = 9;
% sf = 4; nlevel = 20; kerneltype = 1; kernelwidth = 3.2;

% For real image 'flowers', some examples of degradation setting are given as follows.
% sf = 2; nlevel = 60; kerneltype = 1; kernelwidth = 1.2;
% sf = 2; nlevel = 60; kerneltype = 3; nk          = 4;
% sf = 3; nlevel = 60; kerneltype = 1; kernelwidth = 2.4;
% sf = 3; nlevel = 60; kerneltype = 3; nk          = 6;
% sf = 4; nlevel = 60; kerneltype = 1; kernelwidth = 3;

% For real image 'stars', some examples of degradation setting are given as follows.
% sf = 2; nlevel = 20; kerneltype = 1; kernelwidth = 0.8;
% sf = 2; nlevel = 20; kerneltype = 3; nk          = 4;
% sf = 3; nlevel = 20; kerneltype = 1; kernelwidth = 1.2;
% sf = 3; nlevel = 20; kerneltype = 3; nk          = 4;
% sf = 4; nlevel = 20; kerneltype = 1; kernelwidth = 1.6;

% For real image sets 'Set5','Set14','BSD100','Urban100', some examples of degradation are:
% sf = 2; nlevel = 10; kerneltype = 1; kernelwidth = 0.4;
% sf = 3; nlevel = 10; kerneltype = 1; kernelwidth = 0.8;
% sf = 4; nlevel = 10; kerneltype = 1; kernelwidth = 1.2;

%%=======  ======= ======= ======= ======= ======= ======= ======= ======= =======


%% select testing dataset, use GPU or not, ...
setTest      = imageSets([1]); %
showResult   = 1; % 1, show images; 2, save restored images
pauseTime    = 1;
useGPU       = 1; % 1 or 0, true or false
method       = 'SRMD';
folderTest   = 'testsets';
folderResult = 'results';
if ~exist(folderResult,'file')
    mkdir(folderResult);
end

%% scale factor (2, 3, 4)

sf          = 4; %{2, 3, 4}

%% load model with scale factor sf
folderModel = 'models';
load(fullfile(folderModel,['SRMDx',int2str(sf),'.mat']));
%net.layers = net.layers(1:end-1);
net = vl_simplenn_tidy(net);
if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

%% degradation parameter (noise level and kernel) setting
%############################# noise level ################################
% noise level, from a range of [0, 75]

nlevel     = 10;  % [0, 75]

kerneltype = 1;   % {1, 2, 3}

%############################### kernel ###################################
% there are tree types of kernels, including isotropic Gaussian,
% anisotropic Gaussian, and estimated kernel k_b for isotropic Gaussian k_d
% under direct downsampler (x2 and x3 only).

if kerneltype == 1
    % type 1, isotropic Gaussian---although it is a special case of anisotropic Gaussian.
    kernelwidth = 1.7; % from a range of [0.2, 2] for sf = 2, [0.2, 3] for sf = 3, and [0.2, 4] for sf = 4.
    kernel = fspecial('gaussian',15, kernelwidth); % Note: the kernel size is fixed to 15X15.
    tag    = ['_',method,'_Real_x',num2str(sf),'_itrG_',int2str(kernelwidth*10),'_nlevel_',int2str(nlevel)];
    
elseif kerneltype == 2
    % type 2, anisotropic Gaussian
    nk     = 1;      % randi(size(net.meta.AtrpGaussianKernel,4)); %  select one
    kernel = net.meta.AtrpGaussianKernel(:,:,:,nk);
    tag    = ['_',method,'_Real_x',num2str(sf),'_atrG_',int2str(nk),'_nlevel_',int2str(nlevel)];
    
elseif kerneltype == 3 && ( sf==2 || sf==3 )
    % type 3, estimated kernel k_b (x2 and x3 only)
    nk     = 5;      %randi(size(net.meta.directKernel,4)); % select one
    kernel = net.meta.directKernel(:,:,:,nk);
    tag    = ['_',method,'_Real_x',num2str(sf),'_dirG_',int2str(nk),'_nlevel_',int2str(nlevel)];
end

%##########################################################################

surf(kernel) % show kernel
view(45,55);
title('Assumed kernel');
xlim([1 15]);
ylim([1 15]);
pause(2)
close;

%% for degradation maps
global degpar;
degpar = single([net.meta.P*kernel(:); nlevel(:)/255]);


for n_set = 1 : numel(setTest)
    
    %% search images
    setTestCur = cell2mat(setTest(n_set));
    disp('--------------------------------------------');
    disp(['    ----',setTestCur,'-----Super-Resolution-----']);
    disp('--------------------------------------------');
    folderTestCur = fullfile(folderTest,setTestCur);
    ext                 =  {'*.jpg','*.png','*.bmp'};
    filepaths           =  [];
    for i = 1 : length(ext)
        filepaths = cat(1,filepaths,dir(fullfile(folderTestCur, ext{i})));
    end
    
    %% prepare results
    folderResultCur = fullfile(folderResult, [setTestCur,tag]);
    if ~exist(folderResultCur,'file')
        mkdir(folderResultCur);
    end
    
    %% perform SISR
    for i = 1 : length(filepaths)
        
        LR  = imread(fullfile(folderTestCur,filepaths(i).name));
        C   =  size(LR,3);
        if C == 1
            LR = cat(3,LR,LR,LR);
        end
        [~,imageName,ext] = fileparts(filepaths(i).name);
        input    = im2single(LR);
        %tic
        if useGPU
            input = gpuArray(input);
        end
        res = vl_srmd(net, input,[],[],'conserveMemory',true,'mode','test','cudnn',true);
        %res = vl_srmd_concise(net, input); % a concise version of "vl_srmd".
        %res = vl_srmd_matlab(net, input); % you should also set "useGPU = 0;" and comment "net = vl_simplenn_tidy(net);"
        
        output_RGB = gather(res(end).x);
        
        %toc;
        
        disp([setTestCur,'    ',int2str(i),'    ','    ',filepaths(i).name]);
        
        if showResult
            imshow(cat(2,imresize(im2uint8(LR),sf),im2uint8(output_RGB)));
            drawnow;
            title(['SRMD     ',filepaths(i).name],'FontSize',12)
            pause(pauseTime)
            imwrite(output_RGB,fullfile(folderResultCur,[imageName,'_x',int2str(sf),'.png']));% save results
            
        end
    end
end



























