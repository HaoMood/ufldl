function [Z, V] = zca2(X)
epsilon = 1e-4;
% You should be able to use the code from your PCA/ZCA exercise
% Retain all of the components from the ZCA transform (i.e. do not do
% dimensionality reduction)

% x is the input patch data of size
% z is the ZCA transformed data. The dimenison of z = x.

%%% YOUR CODE HERE %%%
Z = zeros(size(X));
[n, m] = size(X);
Mu = mean(X);
X = X - ones(n, 1) * Mu;
Sigma = X * X' / m;
[U, S, ~] = svd(Sigma);
Z = U * diag(1 ./ sqrt(diag(S))) * U' * X;
V = U * diag(1 ./ sqrt(diag(S))) * U';

