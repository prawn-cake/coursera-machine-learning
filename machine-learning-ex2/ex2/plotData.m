function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
sizes = zeros(2, 1);

% Determin sizes of new matrixes
for i = 1:length(y)
    if y(i) == 0
        sizes(1) += 1
    else
        sizes(2) += 1
end

N = zeros(sizes(1), 2);
P = zeros(sizes(2), 2);

% track current indexes
p_idx = 1;
n_idx = 1;
for i = 1:length(y)
    if y(i) == 0
        N(n_idx, :) = X(i, :);
        n_idx += 1;
    else
        P(p_idx, :) = X(i, :);
        p_idx += 1;
    endif
end


% Plot both type of markers
plot(P(:, 1), P(:, 2), 'k+', N(:, 1), N(:, 2), 'ko', "markersize", 10);

% Positive markers
%plot(P(:, 1), P(:, 2), 'k+', "color", 'g', "markersize", 10);



% =========================================================================



hold off;

end
