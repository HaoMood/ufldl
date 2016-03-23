function convolvedFeatures = cnnConvolve(filterDim, numFilters, images, W, b)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  filterDim - filter (feature) dimension
%  numFilters - number of feature maps
%  images - large images to convolve with, matrix in the form
%           images(r, c, image number)
%  W, b - W, b for features from the sparse autoencoder
%         W is of shape (filterDim,filterDim,numFilters)
%         b is of shape (numFilters,1)
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)

numImages = size(images, 3);
m = numImages;
imageDim = size(images, 1);
R = imageDim;
C = size(images, 2);
BW = filterDim;
convDim = R - BW + 1;
n2 = numFilters;

convolvedFeatures = zeros(R-BW+1, R-BW+1, n2, m);
%convolvedFeatures = zeros(convDim, convDim, numFilters, numImages);

% Instructions:
%   Convolve every filter with every image here to produce the 
%   (imageDim - filterDim + 1) x (imageDim - filterDim + 1) x numFeatures x numImages
%   matrix convolvedFeatures, such that 
%   convolvedFeatures(imageRow, imageCol, featureNum, imageNum) is the
%   value of the convolved featureNum feature for the imageNum image over
%   the region (imageRow, imageCol) to (imageRow + filterDim - 1, imageCol + filterDim - 1)
%
% Expected running times: 
%   Convolving with 100 images should take less than 30 seconds 
%   Convolving with 5000 images should take around 2 minutes
%   (So to save time when testing, you should convolve with less images, as
%   described earlier)


%for imageNum = 1:numImages
  %for filterNum = 1:numFilters
for i = 1: m
    for k = 1: n2

    % convolution of image with feature matrix
    %convolvedImage = zeros(convDim, convDim);
    % s2 = zeros(R-BW+1, R-BW+1);

    % Obtain the feature (filterDim x filterDim) needed during the convolution
    %%% YOUR CODE HERE %%%
    % filter = W(:, :, k);

    % Flip the feature matrix because of the definition of convolution, as explained later
    % filter = rot90(squeeze(filter),2);
      
    % Obtain the image
    %im = squeeze(images(:, :, imageNum));
    % xi = squeeze(images(:, :, i));

    % Convolve "filter" with "im", adding the result to convolvedImage
    % be sure to do a 'valid' convolution
    % Add the bias unit

    %%% YOUR CODE HERE %%%
    % s2 = conv2(squeeze(images(:, :, i)), rot90(squeeze(W(:, :, k)), 2), 'valid') + b(k);
    s2 = conv2(images(:, :, i), rot90(W(:, :, k), 2), 'valid') + b(k);
%    s2 = conv2(xi, filter, 'valid') + b(k);
    
    % Then, apply the sigmoid function to get the hidden activation

    %%% YOUR CODE HERE %%%
    a2 = 1 ./ (1 + exp(-s2));

    convolvedFeatures(:, :, k, i) = a2;
 %   convolvedFeatures(:, :, filterNum, imageNum) = convolvedImage;
  end
end


end

