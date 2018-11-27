
function net = model_init_SRMD_x3

sf    = 3;

lr11  = [1 1];
lr10  = [1 0];
weightDecay = [1 1];
nCh = 128; % number of channels
C   = 3;   % C = 3 for color image, C = 1 for gray-scale image
dim_PCA  = 15;

useBnorm  = 1; % if useBnorm  = 0, you should also use adam.

% Define network
net.layers = {} ;

%net.layers{end+1} = struct('type', 'SubP','scale',1/2) ;

net.layers{end+1} = struct('type', 'concat') ;

net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{orthrize(sqrt(2/(9*nCh))*randn(3,3,C+dim_PCA+1,nCh,'single')), zeros(nCh,1,'single')}}, ...
    'stride', 1, ...
    'pad', 1, ...
    'dilate',1, ...
    'learningRate',lr11, ...
    'weightDecay',weightDecay, ...
    'opts',{{}}) ;
net.layers{end+1} = struct('type', 'relu','leak',0) ;

for i = 1:1:10
    
    if useBnorm ~= 0
        net.layers{end+1} = struct('type', 'conv', ...
            'weights', {{orthrize(sqrt(2/(9*nCh))*randn(3,3,nCh,nCh,'single')), zeros(nCh,1,'single')}}, ...
            'stride', 1, ...
            'learningRate',lr10, ...
            'dilate',1, ...
            'weightDecay',weightDecay, ...
            'pad', 1, 'opts', {{}}) ;
        net.layers{end+1} = struct('type', 'bnorm', ...
            'weights', {{clipping(sqrt(2/(9*nCh))*randn(nCh,1,'single'),0.01), zeros(nCh,1,'single'),[zeros(nCh,1,'single'), 0.01*ones(nCh,1,'single')]}}, ...
            'learningRate', [1 1 1], ...
            'weightDecay', [0 0]) ;
    else
        net.layers{end+1} = struct('type', 'conv', ...
            'weights', {{orthrize(sqrt(2/(9*nCh))*randn(3,3,nCh,nCh,'single')), zeros(nCh,1,'single')}}, ...
            'stride', 1, ...
            'learningRate',lr11, ...
            'dilate',1, ...
            'weightDecay',weightDecay, ...
            'pad', 1, 'opts', {{}}) ;
    end
    net.layers{end+1} = struct('type', 'relu','leak',0) ;
    
end

net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{orthrize(sqrt(2/(9*nCh))*randn(3,3,nCh,sf^2*C,'single')), zeros(sf^2*C,1,'single')}}, ...
    'stride', 1, ...
    'learningRate',lr10, ...
    'dilate',1, ...
    'weightDecay',weightDecay, ...
    'pad', 1, 'opts', {{}}) ;
net.layers{end+1} = struct('type', 'SubP','scale',sf) ;

net.layers{end+1} = struct('type', 'loss') ; % make sure the new 'vl_nnloss.m' is in the same folder.

% Fill in default values
net = vl_simplenn_tidy(net);

end


function W = orthrize(a)

s_ = size(a);
a = reshape(a,[size(a,1)*size(a,2)*size(a,3),size(a,4),1,1]);
[u,d,v] = svd(a, 'econ');
if(size(a,1) < size(a, 2))
    u = v';
end
%W = sqrt(2).*reshape(u, s_);
W = reshape(u, s_);

end


function A = clipping2(A,b)

A(A<b(1)) = b(1);
A(A>b(2)) = b(2);

end



function A = clipping(A,b)

A(A>=0&A<b) = b;
A(A<0&A>-b) = -b;

end





