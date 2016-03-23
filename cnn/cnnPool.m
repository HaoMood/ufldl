function a3 = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

a2 = convolvedFeatures;
[R2, C2, n2, m] = size(a2);
BW2 = poolDim;

% numImages = size(convolvedFeatures, 4);
% numFilters = size(convolvedFeatures, 3);
% convolvedDim = size(convolvedFeatures, 1);

a3 = zeros(R2 / BW2,  C2 / BW2, n2, m);
n3 = n2;
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

%%% YOUR CODE HERE %%%
W2 = ones(BW2, BW2) / (BW2 * BW2);
for i = 1: m
    for k = 1: n3
         s3 = conv2( a2(:, :, k, i), rot90(W2, 2), 'valid' );
         a3(:, :, k, i) = s3(1: BW2: end, 1: BW2: end);
    end
end

