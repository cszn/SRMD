%==========================================================================
% This is the testing code of SRMD for the <general degradation> of SISR.

%==========================================================================
% The basic idea of SRMD is to learn a CNN to infer the MAP of general SISR, i.e.,
% solve x^ = arg min_x 1/(2 sigma^2) ||(kx)\downarrow_s - y||^2 + lamda \Phi(x)
% via x^ = CNN(y,k,sigma;\Theta) or HR^ = CNN(LR,kernel,noiselevel;\Theta).
%
% There involves two important factors, i.e., blur kernel (k; kernel) and noise
% level (sigma; nlevel).
%
% For more information, please refer to the following paper.
%
% @inproceedings{zhang2018learning,
%   title={Learning a single convolutional super-resolution network for multiple degradations},
%   author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
%   booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
%   pages={3262-3271},
%   year={2018}
% }
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
addpath('kernels');
imageSets    = {'Set5','Set14','BSD100','Urban100'}; % testing dataset

%% select testing dataset, use GPU or not, ...
setTest      = imageSets([1]); %
showResult   = 1; % 1, show ground-truth, bicubicly interpolated LR image, and restored HR images by SRMD; 2, save restored images
pauseTime    = 1;
useGPU       = 1; % 1 or 0, true or false
method       = 'SRMD';
folderTest   = 'testsets';
folderResult = 'results';
if ~exist(folderResult,'file')
    mkdir(folderResult);
end

%% scale factor (2, 3, 4)

sf          = 3; %{2, 3, 4}

%% load model with scale factor sf
folderModel = fullfile('data',['SRMD_x',num2str(sf)]);
load(fullfile(folderModel,['SRMD_x',num2str(sf),'-epoch-1.mat'])); % load the trained model

net.layers = net.layers(1:end-1);
net = vl_simplenn_tidy(net);
if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

%% degradation parameter (noise level and kernel) setting
%############################# noise level ################################
% noise level, from a range of [0, 75]

nlevel     = 15;  % [0, 75]

kerneltype = 1;  % {1, 2, 3}

%############################### kernel ###################################

ksize = 15;
theta = pi*rand(1);
l1    = 0.1+9.9*rand(1);
l2    = 0.1+(l1-0.1)*rand(1);
kernel =  anisotropic_Gaussian(ksize,theta,l1,l2);
tag    = ['_',method,'_x',num2str(sf)];

%##########################################################################

surf(kernel) % show kernel
view(45,55);
title('Assumed kernel');
xlim([1 15]);
ylim([1 15]);
pause(2)
close;

%% for degradation maps
load('PCA_P.mat')
global degpar;
degpar = single([P*kernel(:); nlevel(:)/255]);


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
    eval(['PSNR_',setTestCur,'_x',num2str(sf),' = zeros(length(filepaths),1);']);
    eval(['SSIM_',setTestCur,'_x',num2str(sf),' = zeros(length(filepaths),1);']);
    folderResultCur = fullfile(folderResult, [setTestCur,tag]);
    if ~exist(folderResultCur,'file')
        mkdir(folderResultCur);
    end
    
    %% perform SISR
    for i = 1 : length(filepaths)
        
        HR  = imread(fullfile(folderTestCur,filepaths(i).name));
        C   = size(HR,3);
        if C == 1
            HR = cat(3,HR,HR,HR);
        end
        [~,imageName,ext] = fileparts(filepaths(i).name);
        HR  = modcrop(HR, sf);
        label_RGB = HR;
        blury_HR = imfilter(im2double(HR),double(kernel),'replicate'); % blur
        LR       = imresize(blury_HR,1/sf,'bicubic'); % bicubic downsampling
        randn('seed',0);
        LR_noisy = LR + nlevel/255.*randn(size(LR)); % add random noise (AWGN)
        input    = single(LR_noisy);
        %tic
        if useGPU
            input = gpuArray(input);
        end
        res = vl_srmd(net, input,[],[],'conserveMemory',true,'mode','test','cudnn',true);
        %res = vl_srmd_concise(net, input); % a concise version of "vl_srmd".
        %res = vl_srmd_matlab(net, input); % When use this, you should also set "useGPU = 0;" and comment "net = vl_simplenn_tidy(net);"
        
        output_RGB = gather(res(end).x);
        
        %toc;
        if C == 1
            label  = mean(im2double(HR),3);
            output = mean(output_RGB,3);
        else
            label  = rgb2ycbcr(im2double(HR));
            output = rgb2ycbcr(double(output_RGB));
            label  = label(:,:,1);
            output = output(:,:,1);
        end
        
        %% calculate PSNR and SSIM
        [PSNR_Cur,SSIM_Cur] = Cal_PSNRSSIM(label*255,output*255,sf,sf); %%% single
        disp([setTestCur,'    ',int2str(i),'    ',num2str(PSNR_Cur,'%2.2f'),'dB','    ',filepaths(i).name]);
        eval(['PSNR_',setTestCur,'_x',num2str(sf),'(',num2str(i),') = PSNR_Cur;']);
        eval(['SSIM_',setTestCur,'_x',num2str(sf),'(',num2str(i),') = SSIM_Cur;']);
        if showResult
            imshow(cat(2,label_RGB,imresize(im2uint8(LR_noisy),sf),im2uint8(output_RGB)));
            drawnow;
            title(['SISR     ',filepaths(i).name,'    ',num2str(PSNR_Cur,'%2.2f'),'dB'],'FontSize',12)
            pause(pauseTime)
            imwrite(output_RGB,fullfile(folderResultCur,[imageName,'_x',int2str(sf),'_',int2str(PSNR_Cur*100),'.png']));% save results
        end
        
    end
    disp(['Average PSNR is ',num2str(mean(eval(['PSNR_',setTestCur,'_x',num2str(sf)])),'%2.2f'),'dB']);
    disp(['Average SSIM is ',num2str(mean(eval(['SSIM_',setTestCur,'_x',num2str(sf)])),'%2.4f')]);
    
    %% save PSNR and SSIM results
    save(fullfile(folderResultCur,['PSNR_',setTestCur,'_x',num2str(sf),'.mat']),['PSNR_',setTestCur,'_x',num2str(sf)]);
    save(fullfile(folderResultCur,['SSIM_',setTestCur,'_x',num2str(sf),'.mat']),['SSIM_',setTestCur,'_x',num2str(sf)]);
    
end



























