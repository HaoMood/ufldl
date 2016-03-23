function [J, grad, pred_prob] = supervised_dnn_cost( para, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(para, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);

%% forward prop
%%% YOUR CODE HERE %%%
Theta = cell(numel(stack), 1);
L = numel(ei.layer_sizes)+1;
for d = 1: L-1
    Theta{d} = [stack{d}.b stack{d}.W];
end

X = data';
n = cell(L, 1);
[m, n{1}] = size(X); 
n{L} = 10;
X = [ones(m, 1) X];
y = labels';

A = cell(L, 1);
S = cell(L, 1);
A{1} = X';
S{1} = [];
for l = 2: L-1
   S{l} = Theta{l-1} * A{l-1};
   n{l} = size(S{l}, 1);
   A{l} = g(S{l});
   A{l} = [ones(1, m); A{l}];
end

A{L} = exp(Theta{L-1} * A{L-1}) ./ (ones(n{L}, 1) * sum(exp(Theta{L-1} * A{L-1})));
pred_prob = A{L};

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  J = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
J = sum(log(sum(exp(Theta{L-1}*A{L-1}))), 2) - sum(sum( Theta{L-1}(y, :)' .* A{L-1} ), 2);

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
Delta = cell(L, 0);
Delta{1} = [];

K = 10;
sub = zeros(K, m);
idx = sub2ind(size(sub), y, 1: m);
sub(idx) = 1;
Delta{L} = A{L} - sub;

gradStack_tmp = cell(L, 1);
gradStack_tmp{L-1} = Delta{L} * A{L-1}';
gradStack{L-1}.b = gradStack_tmp{L-1}(:, 1);
gradStack{L-1}.W = gradStack_tmp{L-1}(:, 2: end);

for l = L - 1: 2
    Delta{l} = Theta{l}' * Delta{l+1} .* (A{l} .* (1-A{l}));
    Delta{l} = Delta{l}(2: end, :);
    gradStack_tmp{l-1} = Delta{l} * A{l-1}';
    gradStack{l-1}.b = gradStack_tmp{l-1}(:, 1);
    gradStack{l-1}.W = gradStack_tmp{l-1}(:, 2: end);
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
% for l = 1: L - 1
%     grad = [grad; gradStack{l}(:)];
% end