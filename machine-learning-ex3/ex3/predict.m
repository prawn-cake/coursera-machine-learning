function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% X here is X(i, :), size(X) --> 1 x 400

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.

% Produce new feature x0
printf('size(X) = %i,%i; m=%i \n', size(X, 1), size(X, 2), m);
X = [ones(size(X), 1), X];

% Keeping 5000 units as row values
z2 = X * Theta1';
% 5000 x 25
a2 = sigmoid(z2);

% Produce bias unit for layer 3
% NOTE: bias unit is a new feature
a2 = [ones(size(a2, 1), 1) a2];
z3 = a2 * Theta2';
h_x = sigmoid(z3);

% h_x is our result vector
% Got some help in discussion: https://www.coursera.org/learn/machine-learning/programming/Y54Zu/multi-class-classification-and-neural-networks/discussions/T-Pbwv9aEeS16yIACyoj1Q
[val, idx] = max(h_x, [], 2);
p = idx;


% =========================================================================


end
