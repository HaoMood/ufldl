function [J, grad] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m = size(X,2);
  n = size(X,1);
  n = n - 1;

  % theta is a vector;  need to reshape to n x num_classes.
  Theta = reshape(theta, n + 1, []);
  Theta = [Theta zeros(n+1, 1)];
  K = size(Theta,2);
  
  % initialize objective value and gradient.
  J = 0;
  grad = zeros(size(Theta(:, 1: end-1)));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  X = X';
  y = y';
  J = sum(log(sum(exp(X * Theta), 2) + 1)) - sum(sum( X' .* Theta(:, y') ));
  
  for k = 1: K - 1 
      grad(:, k) = X' * ((y == k) .*  (exp(X * Theta(:, k)) ./ sum(exp(X * Theta), 2) - 1));
  end

  grad=grad(:); % make gradient a vector for minFunc

