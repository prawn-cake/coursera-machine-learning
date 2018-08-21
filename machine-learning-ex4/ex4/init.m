% Init parameters
load('ex4data1.mat');
load('ex4weights.mat');
nn_params = [Theta1(:) ; Theta2(:)];
lambda = 0;
input_layer_size  = 400;  % 20x20 Input Images of Digits↲
hidden_layer_size = 25;   % 25 hidden units↲
num_labels = 10;

% Feed-forwarding
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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

%grad = (1 / m) .* X' * (h_x - y_matrix);
J = (1 / m) * sum( sum((-y_matrix .* log(h_x) - (1 - y_matrix) .* log(1 - h_x))) );

th1 = Theta1(:, 2:end);
th2 = Theta2(:, 2:end);
reg_term = (lambda / (2 * m)) * ( sum( sum(th1 .^ 2) ) + sum( sum(th2 .^ 2) ) );

J = J + reg_term;

% Backpropagation

