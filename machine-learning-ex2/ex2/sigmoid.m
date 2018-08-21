function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

g = 1 ./ (1 + exp(-z))

% Don't need it, but good to know
%if (ismatrix(z))
%    % pass
%    g = 1 / (1 + exp(-z))
%elseif (isvector(z))
%    % pass
%    g = 1 / (1 + exp(-z))
%else
%    % for scalar
%    g = 1 / (1 + exp(-z))
%endif



% =============================================================

end
