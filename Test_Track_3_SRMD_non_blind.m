% http://www.vision.ee.ethz.ch/en/ntire18/
% paper: http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Timofte_NTIRE_2018_Challenge_CVPR_2018_paper.pdf

% 1) non-blind SRMD can handle Track 2, 3 and 4 in a single model.
% 2) non-blind SRMD can produce good results with accurate blur kernel of LR images.
% Since non-blind SRMD also takes the blur kernel (degradation maps) as input, we use the information of LR image in Track 1 to facilitate the blur kernel estimation.

% In this code, the dimention-reduced blur kernels are precalculated and
% are stored in `kernel_reduced_3`of the `SRMD_non_blind.mat`.

% Note: we use a single `SRMD_non_blind.mat` model in `Test_Track_3_SRMD_non_blind.m` and `Test_Track_4_SRMD_non_blind.m`.

gpu         = 1;

%% load model
load(fullfile('model','SRMD_non_blind.mat'));

if gpu
    net = vl_simplenn_move(net, 'gpu') ;
end

%% LR images
folderLR       = 'H:\matlabH\DIV2K_test_LR_difficult';

folderResultCur= 'Results_Track_3_non_blind';
if ~isdir(folderResultCur)
    mkdir(folderResultCur)
end

global kncf;

for i = 1:100
    
    Iname = num2str(i+900,'%04d');
    LR = im2single(imread(fullfile(folderLR,[Iname,'x4d.png'])));
        
    kncf     = kernel_reduced_3(:,i); % reduced blur kernel after PCA projection
    
    tic;
    if gpu
        input = gpuArray(single(LR));
    end
    res = vl_simplenn(net, input,[],[],'conserveMemory',true,'mode','test');
    im =  res(end).x;
    if gpu
        im = gather(im);
    end
    toc;
    imshow(cat(2,imresize(LR,4),im));
    imwrite(im, fullfile(folderResultCur,[Iname,'x4d.png']));
    pause(0.001)
    
end
