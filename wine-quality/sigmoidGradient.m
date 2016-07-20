function g = sigmoidGradient(z)
##  g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
##    evaluated at z. Used in back propagation gradient descent.

  g = sigmoid(z) .* (1 - sigmoid(z));

end
