function J = computeLoss(X, y, theta)
  ## J = computeLoss(X, y, theta) computes the cost of using theta as the
  ## parameter for linear regression to fit the data points in X and y

  m = length(y);
  J = (1/(2*m)) * sum(((X * theta) - y) .^ 2);
  # J = (1/(2*m)) * (X * theta - y)' * (X * theta - y);
end
