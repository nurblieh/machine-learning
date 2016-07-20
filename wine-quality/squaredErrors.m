function J = squaredErrors(p, y)
  ## J = squaredErrors(p, y) compute Mean Squared Errors given p predictions
  ## and y real values. Useful for linear regression cost.

  m = length(y);
  ## Note, real MSE doesn't have 1/2 as below, but it's more useful for us.
  J = (1/(2*m)) * sum((p - y) .^ 2);
end
