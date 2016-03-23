function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)

if ~exist('pred','var')
    pred = false;
end;

imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

D = images;
a1 = D;
K = numClasses;
n4 = K;
m = numImages;
r1 = imageDim;
c1 = r1;
tau1 = filterDim;
tau2 = poolDim;
n2 = numFilters;
y = labels;

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

W1 = Wc;
b1 = bc;
W3 = Wd;
b3 = bd;

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

W1_grad = Wc_grad;
b1_grad = bc_grad;
W3_grad = Wd_grad;
b3_grad = bd_grad;

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

r2 = convDim;
c2 = r2;
r3 = r2 / tau2;
c3 = c2 / tau2;
n3 = n2;

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = zeros(convDim,convDim,numFilters,numImages);

a2 = activations;

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

a3 = activationsPooled;

%%% YOUR CODE HERE %%%
W2 = ones(tau2, tau2) / (tau2^2);
for i = 1: m
    for k  = 1: n2
         a2(:, :, k, i) = 1 ./ (1 + exp( -conv2(a1(:, :, i), rot90(W1(:, :, k), 2), 'valid') - b1(k)));
         s3 = conv2(a2(:, :, k, i), rot90(W2, 2), 'valid');
         a3(:, :, k, i) = s3(1: tau2: end, 1: tau2: end);
    end
end


% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
a3 = reshape(a3, [], m);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(K, m);


%%% YOUR CODE HERE %%%
h = exp(W3 * a3 + b3 * ones(1, m)) ./ ( ones(K, 1) * sum(exp(W3 * a3 + b3 * ones(1, m))) );
probs = h;

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost

%%% YOUR CODE HERE %%%
idx = sub2ind(size(h), y', 1: m);
J = -sum(log(h(idx)));
cost = J;
% cost = sum(log(sum(exp(W3 * a3 + b3 * ones(1, m)))), 2) - sum(sum( W3(y, :)' .* a3 ) + b3(y, :)', 2);

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

%%% YOUR CODE HERE %%%
sub = zeros(size(h));
sub(idx) = 1;

delta4 = h - sub;

W3_grad = delta4 * a3';
b3_grad = sum(delta4, 2);

delta3 = W3' * delta4;
delta3 = reshape(delta3, r3, c3, n3, m);

delta2 = zeros(r2, c2, n2, m);

for i = 1: m
    for k = 1: n2  
         delta2(:, :, k, i) = kron(delta3(:, :, k, i), ones(tau2, tau2)) .* a2(:, :, k, i) .* (1-a2(:, :, k, i)) / (tau2^2);

         W1_grad(:, :, k) = W1_grad(:, :, k) + conv2(D(:, :, i), rot90(delta2(:, :, k, i), 2), 'valid');
         b1_grad(k) = b1_grad(k) + sum(sum(delta2(:, :, k, i)));
    end
end

%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

%%% YOUR CODE HERE %%%


%% Unroll gradient into grad vector for minFunc
grad = [W1_grad(:) ; W3_grad(:) ; b1_grad(:) ; b3_grad(:)];

end
