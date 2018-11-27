


%  ksize - e.g., 15, kernel size
%  theta - [0,  pi], rotation angle range
%  l1    - [0.1,10], scaling of eigenvalues
%  l2    - [0.1,l1], scaling of eigenvalues

addpath('kernels');
format compact
ksize       = 15;    % kernel size
num_samples = 10000; % number of sampled kernels
dim_PCA     = 15;

for i = 1:num_samples
    if mod(i,1000)==0
        disp(i);
    end
    theta = pi*rand(1);
    l1    = 0.1+9.9*rand(1);
    l2    = 0.1+(l1-0.1)*rand(1);
    % l2 = l1; % you will get isotropic Gaussian kernel
    
    kernel =  anisotropic_Gaussian(ksize,theta,l1,l2);
    vec_kernels(:,i) = kernel(:);
    
end

% PCA dimensionality reduction
C = double(vec_kernels * vec_kernels');
[V, D] = eig(C);

% perform PCA on features matrix
D = diag(D);
D = cumsum(D) / sum(D);

%k = find(D >= 2e-3, 1); % ignore 0.2% energy
k = ksize^2 - dim_PCA + 1;

% choose the largest eigenvectors' projection
V_pca = V(:, k:end);
features_pca = V_pca' * vec_kernels;
P = single(V_pca');

save PCA_P P
