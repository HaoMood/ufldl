%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, X, params)
m = size(X, 2);

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%
% cost = 0;
% Wgrad = zeros(size(W));
% for i = 1: m
%     xi = X(:, i);
%     cost = cost + params.lambda * sum(sqrt((W * xi).^2 + params.epsilon)) + sum((W' * W * xi - xi).^2) / 2;
%     Wgrad = Wgrad + W * (W' * W * xi - xi) * xi' + W * xi * (W' * W * xi - xi)' + params.lambda * W * xi ./ (sqrt((W * xi).^2 + params.epsilon)) * xi';
% end

Wgrad = (W * (W' * W - eye(size(W, 2))) * X  * X'  + W * X * X' * (W' * W - eye(size(W, 2))) + params.lambda * W * X ./ sqrt((W * X).^2 + params.epsilon) * X') / m;
cost = (params.lambda * sum(sum(sqrt((W * X).^2 + params.epsilon))) + 1 / 2 * trace((W' * W - eye(size(W, 2))) * X * X' * (W' * W - eye(size(W, 2))))) / m;

% cost = cost / m;
% Wgrad = Wgrad / m;

% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);