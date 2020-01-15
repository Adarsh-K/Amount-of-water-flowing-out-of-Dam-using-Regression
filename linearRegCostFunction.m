function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% We need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


J=(1/(2*m))*((X*theta-y)'*(X*theta-y))+(lambda/(2*m))*(theta(2:end,:)'*theta(2:end,:));
Theta=theta;
Theta(1,:)=zeros(1,size(theta,2));
grad=(1/m)*(X'*(X*theta-y))+(lambda/m)*Theta;

% =========================================================================

grad = grad(:);

end
