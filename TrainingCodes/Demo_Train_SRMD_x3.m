% This is the training demo of SRMD for scale factor 3.
%
% To run the code, you should install Matconvnet (http://www.vlfeat.org/matconvnet/) first.
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
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com)

%% xxxxxxxxxxxxxxx  Note!  xxxxxxxxxxxxxxx
%
% run 'Demo_Get_PCA_matrix.m' first to calculate the PCA matrix of vectorized
% blur kernels.
%
% ** You should set the training images folders from "generatepatches.m" first. Then you can run "Demo_Train_SRMD_x3.m" directly.
% **
% ** folders    = {'path_of_your_training_dataset'};% set this from "generatepatches.m" first!
% ** stride     = 40*scale;                         % control the number of image patches, from "generatepatches.m"
% ** nimages    = round(length(filepaths));         % control the number of image patches, from "generatepatches.m"
% **
%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

format compact;
addpath('utilities');
addpath('kernels');

global P;
load('PCA_P.mat')

%-------------------------------------------------------------------------
% Configuration
%-------------------------------------------------------------------------

opts.modelName        = 'SRMD_x3'; % model name
opts.learningRate     = [logspace(-4,-4,100),logspace(-4,-4,100)/3,logspace(-4,-4,100)/(3^2),logspace(-4,-4,100)/(3^3),logspace(-4,-4,100)/(3^4)];% you can change the learning rate
opts.batchSize        = 64; % default  
opts.gpus             = [1]; % this code can only support one GPU!
opts.numSubBatches    = 2;
opts.weightDecay      = 0.0005;
opts.expDir           = fullfile('data', opts.modelName);

%-------------------------------------------------------------------------
%  Initialize model
%-------------------------------------------------------------------------

net  = feval(['model_init_',opts.modelName]);

%-------------------------------------------------------------------------
%   Train
%-------------------------------------------------------------------------

[net, info] = model_train(net,  ...
    'expDir', opts.expDir, ...
    'learningRate',opts.learningRate, ...
    'numSubBatches',opts.numSubBatches, ...
    'weightDecay',opts.weightDecay, ...
    'batchSize', opts.batchSize, ...
    'modelname', opts.modelName, ...
    'gpus',opts.gpus) ;





