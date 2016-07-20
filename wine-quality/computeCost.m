function J = computeCost(X, y, theta)
  ## J = computeCost(X, y, theta) computes the cost of using theta as the
  ## parameter for linear regression to fit the data points in X and y

  m = length(y);
  J = squaredErrors(X * theta, y);
end
