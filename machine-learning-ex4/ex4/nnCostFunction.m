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


% Feed-forward propagation

% convert y vector to matrix of [1 0 0 ... ]' of vectors containing only 0 and 1 values
y_matrix = eye(num_labels)(y, :);

% input layer
% X = X;
a1 = [ones(size(X), 1), X];

% hidden layer + add bias
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1) a2];

% S(j+1) = 5000, Sj = 25. Size of the hidden layer is S(j+1) x Sj+1, i.e [25 x 5001]. Check it with assert
%assert(size(a2), [5000 26]);

z3 = a2 * Theta2';

% output layer
a3 = sigmoid(z3);
h_x = a3;

J = (1 / m) * sum( sum((-y_matrix .* log(h_x) - (1 - y_matrix) .* log(1 - h_x))) );

th1 = Theta1(:, 2:end);
th2 = Theta2(:, 2:end);
reg_term = (lambda / (2 * m)) * ( sum( sum(th1 .^ 2) ) + sum( sum(th2 .^ 2) ) );

J = J + reg_term;

% Forward propagation: Done


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

epsilon = 10 ** (-4);

% delta1; size: 5000 x 10

%Delta1 = 0;
%Delta2 = 0;

% Matrix implementation
d3 = a3 - y_matrix;
d2 = (Theta2(:, 2:end)' * d3')' .* sigmoidGradient(z2);
Delta1 = d2' * a1;
Delta2 = d3' * a2;


Theta1_grad = (1 / m) * Delta1;
Theta2_grad = (1 / m) * Delta2;

% Loop implementation
% m is a size of X (number of training examples)
%for i = 1:m
%    % Step 1: perform forward propagation
%    a_1 = X(i, :);
%    a_1 = [1 a_1];
%    z_2 = a_1 * Theta1';
%    a_2 = [1 sigmoid(z_2)];
%    z_3 = a_2 * Theta2';
%    a_3 = sigmoid(z_3);
%
%    % Step 2,3: calculate error delta (error term delta)
%    % d_3: 1 x 10
%    d_3 = a_3 - y_matrix(:, i);
%
%    % Exclude bias unit from Theta2
%    d_2 = Theta2(:, 2:end)' * d_3' .* sigmoidGradient(z_2');
%    % d_2: 1 x 25
%    d_2 = d_2';
%
%    % Step 4: accumulate gradient
%    %d_2 = d_2(2:end);
%
%    % FIXME: wrong values for deltas
%    % NOTE: calc Delta for the layers 1-delta-2 and 2-delta-3
%    Delta1 = Delta1 + d_2 .* a_1';
%    Delta2 = Delta2 + d_3 .* a_2';
%endfor


%Theta_vec = [Theta1(:), Theta2(:)]
% calc Theta1_grad and Theta2_grad

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% NOTE: 1st column shouldn't be regularized because it's a bias unit, so set it to 0
Theta1(:, 1) = 0;
Theta2(:, 1) = 0;

% Calculate regularization term and update gradient
Theta1_grad = Theta1_grad + (lambda / m) * Theta1;
Theta2_grad = Theta2_grad + (lambda / m) * Theta2;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
