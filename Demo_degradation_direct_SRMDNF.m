%==========================================================================
% This is the testing code of SRMDNF for the widely-used degradation with 7x7 
% Gaussian kernel of width 1.6 and direct downsampler of scale factor 3.
% For this degradation, the basic setting is:
%   1. the image blur kernel is; "kernel = fspecial('gaussian',7,1.6)".
%   2. the downsampler is direct downsampler. The scale factor is fixed to 3.
%      You can either train a new model with direct downsamper or use the estimated
%      kernel k_b under direct downsampler.
%   3. there are three SRMDNF models, "SRMDNFx2.mat" for scale factor 2, "SRMDNFx3.mat"
%      for scale factor 3, and "SRMDNFx4.mat" for scale factor 4.
%==========================================================================
% The basic idea of SRMD is to learn a CNN to infer the MAP of general SISR, i.e.,
% solve x^ = arg min_x 1/2 ||(kx)\downarrow_s - y||^2 + lamda \Phi(x)
% via x^ = CNN(y,k;\Theta) or HR^ = CNN(LR,kernel;\Theta).
%
% There involves two important factors, i.e., blur kernel (k; kernel), in SRMDNF.
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
setTest      = imageSets([1,2,3,4]); % select the datasets for each tasks
showResult   = 0; % save restored images
pauseTime    = 0;
useGPU       = 1; % 1 or 0, true or false
method       = 'SRMDNF';
folderTest   = 'testsets';
folderResult = 'results';
if ~exist(folderResult,'file')
    mkdir(folderResult);
end

%% scale factor (3 only)
sf          = 3; % Here, do not set sf to 2 or 4.

%% load model with scale factor sf
folderModel = 'models';
load(fullfile(folderModel,['SRMDNFx',int2str(sf),'.mat']));
%net.layers = net.layers(1:end-1);
net = vl_simplenn_tidy(net);
if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

%% degradation parameter (kernel & noise level) setting
global degpar;
image_kernel =  fspecial('gaussian',7, 1.6); % 7x7 Gaussian kernel k_d with width 1.6
kernel = net.meta.directKernel(:,:,:,5); % corrresponding k_b  under bicubic downsampler
degpar = single(net.meta.P*kernel(:));
tag    = ['_',method,'_x',num2str(sf),'_directG_7_16'];



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
        %% bicubic degradation
        blury_HR = imfilter(im2double(HR),double(image_kernel),'replicate'); % blur
        LR       = imresize(im2double(blury_HR),1/sf,'nearest'); % bicubic downsampling
        input    = im2single(LR);
        %input    = im2single(im2uint8(LR)); % another widely-used setting
        
        if useGPU
            input = gpuArray(input);
        end
        res = vl_srmd(net, input,[],[],'conserveMemory',true,'mode','test','cudnn',true);
        %res = vl_srmd_concise(net, input); % a concise version of "vl_srmd".
        %res = vl_srmd_matlab(net, input); % you should also set "useGPU = 0;" and comment "net = vl_simplenn_tidy(net);"
        
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



























