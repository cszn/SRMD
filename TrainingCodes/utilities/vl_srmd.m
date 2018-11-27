function res = vl_srmd(net, x, dzdy, res, varargin)
%VL_SIMPLENN  Evaluate a SimpleNN network.

%   `Mode`:: `'normal'`
%      Specifies the mode of operation. It can be either `'normal'` or
%      `'test'`. In test mode, dropout and batch-normalization are
%      bypassed. Note that, when a network is deployed, it may be
%      preferable to *remove* such blocks altogether.
%
%   `ConserveMemory`:: `false`
%      Aggressively delete intermediate results. This in practice has
%      a very small performance hit and allows training much larger
%      models. However, it can be useful to disable it for
%      debugging. Keeps the values in `res(1)` (input) and `res(end)`
%      (output) with the outputs of `loss` and `softmaxloss` layers.
%      It is also possible to preserve individual layer outputs
%      by setting `net.layers{...}.precious` to `true`.
%      For back-propagation, keeps only the derivatives with respect to
%      weights.
%
%   `CuDNN`:: `true`
%      Use CuDNN when available.
%
%   `Accumulate`:: `false`
%      Accumulate gradients in back-propagation instead of rewriting
%      them. This is useful to break the computation in sub-batches.
%      The gradients are accumulated to the provided RES structure
%      (i.e. to call VL_SIMPLENN(NET, X, DZDY, RES, ...).
%
%   `BackPropDepth`:: `inf`
%      Limit the back-propagation to top-N layers.
%
%   `SkipForward`:: `false`
%      Reuse the output values from the provided RES structure and compute
%      only the derivatives (backward pass).
%


% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
global degpar;
opts.conserveMemory = true ;
opts.sync = false ;
opts.mode = 'normal' ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.skipForward = false ;
opts.parameterServer = [] ;
opts.holdOn = false ;
opts = vl_argparse(opts, varargin);

n = numel(net.layers) ;
assert(opts.backPropDepth > 0, 'Invalid `backPropDepth` value (!>0)');
backPropLim = max(n - opts.backPropDepth + 1, 1);

if (nargin <= 2) || isempty(dzdy)
    doder = false ;
    if opts.skipForward
        error('simplenn:skipForwardNoBackwPass', ...
            '`skipForward` valid only when backward pass is computed.');
    end
else
    doder = true ;
end

if opts.cudnn
    cudnn = {'CuDNN'} ;
    bnormCudnn = {'NoCuDNN'} ; % ours seems slighty faster
else
    cudnn = {'NoCuDNN'} ;
    bnormCudnn = {'NoCuDNN'} ;
end

switch lower(opts.mode)
    case 'normal'
        testMode = false ;
    case 'test'
        testMode = true ;
    otherwise
        error('Unknown mode ''%s''.', opts. mode) ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
    if opts.skipForward
        error('simplenn:skipForwardEmptyRes', ...
            'RES structure must be provided for `skipForward`.');
    end
    res = struct(...
        'x', cell(1,n+1), ...
        'dzdx', cell(1,n+1), ...
        'dzdw', cell(1,n+1), ...
        'aux', cell(1,n+1), ...
        'stats', cell(1,n+1), ...
        'time', num2cell(zeros(1,n+1)), ...
        'backwardTime', num2cell(zeros(1,n+1))) ;
end

if ~opts.skipForward
    res(1).x = x ;
end

% -------------------------------------------------------------------------
%                                                              Forward pass
% -------------------------------------------------------------------------

for i=1:n
    if opts.skipForward, break; end;
    l = net.layers{i} ;
    res(i).time = tic ;
    switch l.type
        case 'conv'
            res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                'pad', l.pad, ...
                'stride', l.stride, ...
                'dilate', l.dilate, ...
                l.opts{:}, ...
                cudnn{:}) ;
            
        case 'concat'
            if size(degpar,1)~=size(res(i).x,1)
                sigmaMap   = bsxfun(@times,ones(size(res(i).x,1),size(res(i).x,2),1,size(res(i).x,4),'single'),permute(degpar,[3 4 1 2]));
                res(i+1).x = cat(3,res(i).x,sigmaMap);
            else
                res(i+1).x = cat(3,res(i).x,sigmaMap);
            end
            
        case 'convt'
            res(i+1).x = vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
                'crop', l.crop, ...
                'upsample', l.upsample, ...
                'numGroups', l.numGroups, ...
                l.opts{:}, ...
                cudnn{:}) ;
            
        case 'SubP'
            res(i+1).x = vl_nnSubP(res(i).x, [],'scale',l.scale);
            
        case 'loss'
            res(i+1).x = vl_nnloss(res(i).x, l.class) ;
            
        case 'relu'
            if l.leak > 0, leak = {'leak', l.leak} ; else leak = {} ; end
            res(i+1).x = vl_nnrelu(res(i).x,[],leak{:}) ;
            
        case 'sigmoid'
            res(i+1).x = vl_nnsigmoid(res(i).x) ;
            
        case 'bnorm'
            if testMode
                res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
                    'moments', l.weights{3}, ...
                    'epsilon', l.epsilon, ...
                    bnormCudnn{:}) ;
            else
                res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
                    'epsilon', l.epsilon, ...
                    bnormCudnn{:}) ;
            end
            
        case 'custom'
            res(i+1) = l.forward(l, res(i), res(i+1)) ;
            
        otherwise
            error('Unknown layer type ''%s''.', l.type) ;
    end
    
    % optionally forget intermediate results
    needsBProp = doder && i >= backPropLim;
    forget = opts.conserveMemory && ~needsBProp ;
    if i > 1
        lp = net.layers{i-1} ;
        % forget RELU input, even for BPROP
        forget = forget && (~needsBProp || (strcmp(l.type, 'relu') && ~lp.precious)) ;
        forget = forget && ~(strcmp(lp.type, 'loss') || strcmp(lp.type, 'softmaxloss')) ;
        forget = forget && ~lp.precious ;
    end
    if forget
        res(i).x = [] ;
    end
    
    if gpuMode && opts.sync
        wait(gpuDevice) ;
    end
    res(i).time = toc(res(i).time) ;
end

% -------------------------------------------------------------------------
%                                                             Backward pass
% -------------------------------------------------------------------------

if doder
    res(n+1).dzdx = dzdy ;
    for i=n:-1:backPropLim
        l = net.layers{i} ;
        res(i).backwardTime = tic ;
        switch l.type
            
            case 'conv'
                [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                    vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, res(i+1).dzdx, ...
                    'pad', l.pad, ...
                    'stride', l.stride, ...
                    'dilate', l.dilate, ...
                    l.opts{:}, ...
                    cudnn{:}) ;
                
            case 'convt'
                [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                    vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, res(i+1).dzdx, ...
                    'crop', l.crop, ...
                    'upsample', l.upsample, ...
                    'numGroups', l.numGroups, ...
                    l.opts{:}, ...
                    cudnn{:}) ;
                
            case 'bicubic'
                res(i).dzdx = vl_nnbicubic(res(i).x, res(i+1).dzdx, 'scale',l.scale) ;
                        
            case 'SubP'
                res(i).dzdx = vl_nnSubP(res(i).x, res(i+1).dzdx,'scale',l.scale);
                
            case 'loss'
                res(i).dzdx = vl_nnloss(res(i).x, l.class, res(i+1).dzdx) ;
                
            case 'relu'
                if l.leak > 0, leak = {'leak', l.leak} ; else leak = {} ; end
                if ~isempty(res(i).x)
                    res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx, leak{:}) ;
                else
                    % if res(i).x is empty, it has been optimized away, so we use this
                    % hack (which works only for ReLU):
                    res(i).dzdx = vl_nnrelu(res(i+1).x, res(i+1).dzdx, leak{:}) ;
                end
                
           
            case 'bnorm'
                [res(i).dzdx, dzdw{1}, dzdw{2}, dzdw{3}] = ...
                    vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, res(i+1).dzdx, ...
                    'epsilon', l.epsilon, ...
                    bnormCudnn{:}) ;
                % multiply the moments update by the number of images in the batch
                % this is required to make the update additive for subbatches
                % and will eventually be normalized away
                dzdw{3} = dzdw{3} * size(res(i).x,4) ;
                
            case 'pdist'
                res(i).dzdx = vl_nnpdist(res(i).x, l.class, ...
                    l.p, res(i+1).dzdx, ...
                    'noRoot', l.noRoot, ...
                    'epsilon', l.epsilon, ...
                    'aggregate', l.aggregate, ...
                    'instanceWeights', l.instanceWeights) ;
                
            case 'custom'
                res(i) = l.backward(l, res(i), res(i+1)) ;
                
        end % layers
        
        switch l.type
            case {'conv','convt', 'bnorm'}
                if ~opts.accumulate
                    res(i).dzdw = dzdw ;
                else
                    for j=1:numel(dzdw)
                        res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
                    end
                end
                dzdw = [] ;
                if ~isempty(opts.parameterServer) && ~opts.holdOn
                    for j = 1:numel(res(i).dzdw)
                        opts.parameterServer.push(sprintf('l%d_%d',i,j),res(i).dzdw{j}) ;
                        res(i).dzdw{j} = [] ;
                    end
                end
        end
        if opts.conserveMemory && ~net.layers{i}.precious && i ~= n
            res(i+1).dzdx = [] ;
            res(i+1).x = [] ;
        end
        if gpuMode && opts.sync
            wait(gpuDevice) ;
        end
        res(i).backwardTime = toc(res(i).backwardTime) ;
    end
    if i > 1 && i == backPropLim && opts.conserveMemory && ~net.layers{i}.precious
        res(i).dzdx = [] ;
        res(i).x = [] ;
    end
end
