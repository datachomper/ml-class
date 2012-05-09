function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

for i=1: m
	h = sigmoid(theta' * X(i,:)');
	J += -y(i)*log(h) - (1-y(i))*log(1-h);
	for j=1: size(theta)
		grad(j) += ((h-y(i)) * X(i,j));
	end
end
J /= m;
grad = grad ./ m;

norml = 0;
for j=2: length(theta)
	norml += theta(j)^2;
	grad(j) += ((lambda*theta(j))/m);
end
norml *= (lambda/(2*m));

J += norml;

end
