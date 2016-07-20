function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
  ## Performs gradient descent to learn theta
  ##    theta = gradientDescent(X, y, theta, alpha, num_iters) updates theta by 
  ##    taking num_iters gradient steps with learning rate alpha.

  m = length(y);                   # Num samples.
  J_history = zeros(num_iters, 1); # Save our cost over time for graphing.

  for iter = 1:num_iters
    theta = theta - (alpha/m) * (X' * (X * theta - y));
    J_history(iter) = computeCost(X, y, theta);

  endfor

end
