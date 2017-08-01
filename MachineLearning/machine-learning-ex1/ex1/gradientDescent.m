function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    ##  X * theta - m x n * n x 1 matrix, giving m x 1 matrix
    ##  (X * theta) - y gives an m x 1 matrix
    ##  (X * theta) - y)' * X gives 1 x m * m x n, giving 1 x n
    ##  ((X * theta) - y)' * X)' gives n x 1 matrix
    
    newtheta = theta - alpha * (1/m * ((X * theta) - y)' * X)'
    theta = newtheta

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
