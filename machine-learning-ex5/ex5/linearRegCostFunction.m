function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.

h_x = X * theta;
reg_term = lambda / (2 * m) * sum(theta(2:end) .^ 2);
J = 1/(2 * m) * ( sum((h_x - y) .^ 2) ) + reg_term;

num_of_features = size(X, 2);

for n = 1:num_of_features
    if (n == 1)
        % NOTE: don't need to calc reg_term for the theta_0
        grad(n) = (1 / m) * sum((h_x - y) .* X(:, n));
    else
        grad(n) = (1 / m) * sum((h_x - y) .* X(:, n))' + (lambda / m) * theta(n);
    endif
end

%disp(X)
% NOTE: don't need to calc reg_term for the theta_0
%g1 = (1 / m) * sum((h_x - y) .* X(:, 1));
%g2 = (1 / m) * sum((h_x - y) .* X(:, 2:end))' + (lambda / m) * theta(2:end);
%g2 = (1 / m) * sum((h_x - y) .* X(:, 2))' + (lambda / m) * theta(2);
%g2 = (1 / m) * sum((h_x - y) .* X(:, 2))';
%g3 = (1 / m) * sum((h_x - y) .* X(:, 3:end))' + (lambda / m) * theta(3:end);
%g3 = (1 / m) * sum((h_x - y) .* X(:, 3:end))';
%grad = [g1; g2; g3];

% NOTE: it seems to implemented correctly but submit fn doesn't count it




% =========================================================================

grad = grad(:);

end
