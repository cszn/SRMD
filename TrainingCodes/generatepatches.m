function [imdb] = generatepatches

folder     = 'path_of_your_training_dataset';
scale      = 3;

size_label = 32*scale; % size of the HR patch
stride     = 40*scale; % 1) control the total number of patches
stride_low = stride/scale;
batchSize  = 256;
nchannels  = 3;
kernelsize = 15;

size_input = size_label;
padding    = abs(size_input - size_label)/2;

ext               =  {'*.jpg','*.png','*.bmp'};
filepaths         =  [];
for i = 1 : length(ext)
    filepaths = cat(1,filepaths, dir(fullfile(folder, ext{i})));
end

rdis       = randperm(length(filepaths));
nimages    = round(length(filepaths));  % 2) control the total number of patches
scalesc    = min(1,0.5 + 0.05*randi(15,[1,nimages]));
nn         = randi(8,[1,nimages]);

count      = 0;
for i = 1 : nimages
    im = imread(fullfile(folder,filepaths(rdis(i)).name));
    im = data_augmentation(im, nn(i));
    disp([i,nimages,round(count/256)])
    im = imresize(im,scalesc(i),'bicubic');
    
    im = modcrop(im, scale);
    
    LR = ones([size(im,1)/scale,size(im,2)/scale]);
    [hei,wid,~] = size(im);
    for x = 1 : stride : (hei-size_input+1)
        for y = 1 : stride : (wid-size_input+1)
            x_l = stride_low*(x-1)/stride + 1;
            y_l = stride_low*(y-1)/stride + 1;
            if x_l+size_input/scale-1 > size(LR,1) || y_l+size_input/scale-1 > size(LR,2)
                continue;
            end
            count=count+1;
        end
    end
end


numPatches = ceil(count/batchSize)*batchSize;
diffPatches = numPatches - count;
disp([numPatches,numPatches/batchSize,diffPatches]);

disp('---------------PAUSE------------');
%pause

count = 0;
imdb.LRlabels  = zeros(size_label/scale, size_label/scale, nchannels, numPatches,'single');
imdb.HRlabels  = zeros(size_label, size_label, nchannels, numPatches,'single');
imdb.kernels   = zeros(kernelsize, kernelsize, numPatches,'single');
imdb.sigmas    = zeros(1,numPatches,'single');

for i = 1 : nimages
    im = imread(fullfile(folder,filepaths(rdis(i)).name));
    im = data_augmentation(im, nn(i));
    disp([i,nimages,round(count/256)])
    im = imresize(im,scalesc(i),'bicubic');
    
    im = im2double(im);
    
    [LR, HR, kernel, sigma] = degradation_model(im, scale);
    
    [hei,wid,~] = size(HR);

    for x = 1 : stride : (hei-size_input+1)
        for y = 1 : stride : (wid-size_input+1)
            x_l = stride_low*(x-1)/stride + 1;
            y_l = stride_low*(y-1)/stride + 1;
            if x_l+size_input/scale-1 > size(LR,1) || y_l+size_input/scale-1 > size(LR,2)
                continue;
            end
            subim_LR = LR(x_l : x_l+size_input/scale-1, y_l : y_l+size_input/scale-1,1:nchannels);
            subim_HR = HR(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,1:nchannels);
            count=count+1;
            imdb.HRlabels(:, :, :, count) = subim_HR;
            imdb.LRlabels(:, :, :, count) = subim_LR;
            imdb.kernels(:,:,count)       = single(kernel);
            imdb.sigmas(count)            = single(sigma);
            
            if count<=diffPatches
                imdb.LRlabels(:, :, :, end-count+1)   = LR(x_l : x_l+size_input/scale-1, y_l : y_l+size_input/scale-1,1:nchannels);
                imdb.HRlabels(:, :, :, end-count+1)   = HR(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,1:nchannels);
                imdb.kernels(:,:,end-count+1)         = single(kernel);
                imdb.sigmas(end-count+1)              = single(sigma);
            end
        end
    end
end

imdb.set    = uint8(ones(1,size(imdb.LRlabels,4)));


