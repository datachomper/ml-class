function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % Iterate over the number of thetas (features)
    tally = [0;0];
    for j = 1:length(theta)
	for i = 1:m
	    h = (theta' * X(i,:)');
	    tally(j) += (h - y(i)) * X(i,j);
	end
    end
    theta = theta .- (tally .* (alpha/m));

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
