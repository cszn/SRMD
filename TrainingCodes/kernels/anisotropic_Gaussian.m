function [k] = anisotropic_Gaussian(ksize,theta,l1,l2)
% generate an anisotropic Gaussian kernel.
%
% Input:
%  ksize - e.g., 15, kernel size
%  theta - [0,  pi], rotation angle range
%  l1    - [0.1,10], scaling of eigenvalues
%  l2    - [0.1,l1], scaling of eigenvalues

% If l1 = l2, you will get an isotropic Gaussian kernel.

% Output:
%  k     - kernel

v = [cos(theta), -sin(theta); sin(theta), cos(theta)] * [1; 0];
V = [v(1), v(2); v(2), -v(1)];
D = [l1, 0; 0, l2];
Sigma = V * D * V^(-1);
gm    = gmdistribution([0, 0], Sigma);
k     = gm_blur_kernel(gm, ksize);


function [k] = gm_blur_kernel(gm, size)
center = size / 2.0 + 0.5;
k = zeros(size, size);
for y = 1 : size
    for x = 1 : size
        cy = y - center;
        cx = x - center;
        k(y, x) = pdf(gm, [cx, cy]);
    end
end

k = k ./ sum(k(:));


