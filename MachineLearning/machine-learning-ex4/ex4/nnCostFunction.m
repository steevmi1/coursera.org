function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%%  num_labels is K, so this is the inner loop index
%%m = size(y);
%%X = [ones(m, 1) X];

%%  So for every element in the training set, calculate the neural network
%%  outputs, as this is our h_{\Theta}. This gives us a num_labels dimensional
%%  vector for an output. Ultimately, we want to calculate the sum for each
%%  position in the vector.

%%  Does this mean that for each y in our outputs, we need to turn this into a
%%  num_labels-sized vector, with the index for y(i) set to 1?

for i = 1:m
  yvec = zeros(num_labels, 1);
  yvec(y(i)) = 1;
  d3 = zeros(num_labels, 1);
  d1 = zeros(m, 1);
  %%  From backpropagation section in ex4.pdf, moving this here
  a1 = [1 X(i,:)];
  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  a2 = [1 a2];
  z3 = a2 * Theta2';
  a3 = sigmoid(z3);
  
  %%  Now we can maybe compute our cost?
  J = J + (1 / m) * sum(-yvec .* log(a3') - (1 - yvec) .* log(1 .- a3'))';
  
  %%  Compute backpropagation
  d3 = a3' - yvec;
  %%  So now, this next line fails because we try to do element-wise mult
  %%  between a 26x1 vector and a 25x1 vector.
  %%  So I *think* what needs to happen is to add back in the bias to the
  %%  output from the sigmoid gradient, then transpose and that turns it into
  %%  a 1x26 matrix * a 26 x 1 matrix. Then, strip out the basis to turn it
  %%  into a 1x25 one.
  d2 = Theta2' * d3 .* [1 sigmoidGradient(z2)]';
  d2 = d2(2:end);
  %%  Now we have a 1 x 25 matrix. a1 is a 1 x 401 matrix, so d2' * a1 gets us
  %%  back to a 25 x 401 matrix (our original Theta1).
  
  Theta1_grad = Theta1_grad + d2 * a1;
  Theta2_grad = Theta2_grad + d3 * a2;
endfor;

%%  There has to be a way to vectorize this, but I can't seem to guess it.
theta1sum = 0;
for j = 1:size(Theta1,1)
  for k = 2:size(Theta1,2)
    theta1sum = theta1sum + Theta1(j,k)^2;
  endfor;
endfor;

theta2sum = 0;
for j = 1:size(Theta2,1)
  for k = 2:size(Theta2,2)
    theta2sum = theta2sum + Theta2(j,k)^2;
  endfor;
endfor;

J = J + (lambda / (2 * m)) * (theta1sum + theta2sum);

Theta1_grad = (1 / m) .* Theta1_grad + [zeros(size(Theta1, 1), 1) (lambda/m) .* Theta1(:, 2:end)];
Theta2_grad = (1 / m) .* Theta2_grad + [zeros(size(Theta2, 1), 1) (lambda/m) .* Theta2(:, 2:end)];









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
