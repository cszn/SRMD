%==========================================================================
% This is the testing code of SRMDNF for the <general degradation> of SISR.
% For general degradation, the basic setting is:
%   1. there are tree types of kernels, including isotropic Gaussian,
%      anisotropic Gaussian, and estimated kernel k_b for isotropic
%      Gaussian k_d under direct downsampler (x2 and x3 only).
%   3. the downsampler is fixed to bicubic downsampler. 
%      For direct downsampler, you can either train a new model with 
%      direct downsamper or use the estimated kernel k_b under direct 
%      downsampler. The former is preferred.
%   3. there are three models, "SRMDNFx2.mat" for scale factor 2, "SRMDNFx3.mat"
%      for scale factor 3, and "SRMDNFx4.mat" for scale factor 4.
%==========================================================================
% The basic idea of SRMDNF is to learn a CNN to infer the MAP of general SISR, i.e.,
% solve x^ = arg min_x 1/2 ||(kx)\downarrow_s - y||^2 + lamda \Phi(x)
% via x^ = CNN(y,k;\Theta) or HR^ = CNN(LR,kernel;\Theta).
%
% There involves one important factor, i.e., blur kernel (k; kernel), in SRMDNF.
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
imageSets    = {'Set5','Set14','BSD100','Urban100'}; % testing dataset

%% select testing dataset, use GPU or not, ...
setTest      = imageSets([1]); %
showResult   = 1; % 1, show ground-truth, bicubicly interpolated LR image, and restored HR images by SRMD; 2, save restored images
pauseTime    = 1;
useGPU       = 1; % 1 or 0, true or false
method       = 'SRMDNF';
folderTest   = 'testsets';
folderResult = 'results';
if ~exist(folderResult,'file')
    mkdir(folderResult);
end

%% scale factor (2, 3, 4)

sf          = 2;

%% load model with scale factor sf
folderModel = 'models';
load(fullfile(folderModel,['SRMDNFx',int2str(sf),'.mat']));
%net.layers = net.layers(1:end-1);
net = vl_simplenn_tidy(net);
if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

%% degradation parameter (kernel) setting
global degpar;

kerneltype = 1;

%################################kernel####################################
% there are tree types of kernels, including isotropic Gaussian,
% anisotropic Gaussian, and estimated kernel k_b for isotropic Gaussian k_d
% under direct downsampler (x2 and x3 only).
if kerneltype == 1
    % type 1, isotropic Gaussian---although it is a special case of anisotropic Gaussian.
    kernelwidth = 2.6; % from a range of [0.2, 2] for sf = 2, [0.2, 3] for sf = 3, and [0.2, 4] for sf = 4.
    kernel = fspecial('gaussian',15, kernelwidth); % Note: the kernel size is fixed to 15X15.
    tag    = ['_',method,'_x',num2str(sf),'_itrG_',int2str(kernelwidth*10)];
    
elseif kerneltype == 2
    % type 2, anisotropic Gaussian
    nk     = randi(size(net.meta.AtrpGaussianKernel,4)); % randomly select one
    kernel = net.meta.AtrpGaussianKernel(:,:,:,nk);
    tag    = ['_',method,'_x',num2str(sf),'_atrG_',int2str(nk)];
    
elseif kerneltype == 3 && ( sf==2 || sf==3 )
    % type 3, estimated kernel k_b (x2 and x3 only)
    nk     = randi(size(net.meta.directKernel,4)); % randomly select one
    kernel = net.meta.directKernel(:,:,:,nk);
    tag    = ['_',method,'_x',num2str(sf),'_dirG_',int2str(nk)];
    
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
degpar = single(net.meta.P*kernel(:));


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
        input    = single(LR);
        
        if useGPU
            input = gpuArray(input);
        end
        res = vl_srmd(net, input,[],[],'conserveMemory',true,'mode','test','cudnn',true);
        %res = vl_srmd_concise(net, input); % a concise version of "vl_srmd".
        %res = vl_srmd_matlab(net, input); % When use this, you should also set "useGPU = 0;" and comment "net = vl_simplenn_tidy(net);"
        
        output_RGB = gather(res(end).x);
        
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
            imshow(cat(2,label_RGB,imresize(im2uint8(LR),sf),im2uint8(output_RGB)));
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



























