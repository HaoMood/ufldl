function features = feedfowardRICA(filterDim, poolDim, numFilters, images, W)
% feedfowardRICA Returns the convolution of the features given by W with
% the given images. It should be very similar to cnnConvolve.m+cnnPool.m 
% in the CNN exercise, except that there is no bias term b, and the pooling
% is RICA-style square-square-root pooling instead of average pooling.
%
% Parameters:
%  filterDim - filter (feature) dimension
%  numFilters - number of feature maps
%  images - large images to convolve with, matrix in the form
%           images(r, c, image number)
%  W    - W should be the weights learnt using RICA
%         W is of shape (filterDim,filterDim,numFilters)
%
% Returns:
%  features - matrix of convolved and pooled features in the form
%                      features(imageRow, imageCol, featureNum, imageNum)
global params;
numImages = size(images, 3);
imageDim = size(images, 1);
a1 = images;
[r1, c1, m] = size(a1);

tau1 = filterDim;
n2 = numFilters;
convDim = imageDim - filterDim + 1;
r2 = r1 - tau1 + 1;
c2 = c1 - tau1 + 1;
W1 = W;
a2 = zeros(r2, c2, n2, m);

n3 = n2;
tau2 = poolDim;
r3 = r2 / tau2;
c3 = c2 / tau2;

features = zeros(convDim / poolDim, ...
        convDim / poolDim, numFilters, numImages);
a3 = zeros(r3, c3, n3, m);
poolMat = ones(poolDim);
W2 = ones(tau2, tau2);
% Instructions:
%   Convolve every filter with every image just like what you did in
%   cnnConvolve.m to get a response.
%   Then perform square-square-root pooling on the response with 3 steps:
%      1. Square every element in the response
%      2. Sum everything in each pooling region
%      3. add params.epsilon to every element before taking element-wise square-root
%      (Hint: use poolMat similarly as in cnnPool.m)



for i = 1: m
    if mod(i, 500)==0
        fprintf('forward-prop image %d\n', i);
    end
    for k = 1: n2
        % filter = zeros(8,8); % You should replace this
        % Form W, obtain the feature (filterDim x filterDim) needed during the
        % convolution
        %%% YOUR CODE HERE %%%
        % Flip the feature matrix because of the definition of convolution, as explained later
        % filter = rot90(squeeze(filter),2);   
        % Obtain the image
        % im = squeeze(images(:, :, imageNum));
        % resp = zeros(convDim, convDim); % You should replace this
        % Convolve "filter" with "im" to find "resp"
        % be sure to do a 'valid' convolution
        %%% YOUR CODE HERE %%%
        a2(:, :, k, i) = conv2(a1(:, :, i), rot90(W1(:, :, k), 2), 'valid');

        s3 = sqrt(conv2(a2(:, :, k, i).^2, rot90(W2, 2), 'valid') + params.epsilon);
        
        % Then, apply square-square-root pooling on "resp" to get the hidden
        % activation "act"
        % act = zeros(convDim / poolDim, convDim / poolDim); % You should replace this
        %%% YOUR CODE HERE %%%
        % features(:, :, filterNum, imageNum) = act;
        a3(:, :, k, i) = s3(1: tau2: end, 1: tau2: end);
    end
end

features = a3;