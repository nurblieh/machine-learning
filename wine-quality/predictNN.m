function p = predictNN(Theta1, Theta2, X)
  ## p = predictNN(Theta1, Theta2, X) outputs the predicted value of X given the
  ## trained weights of a neural network (Theta1, Theta2)

  m = size(X, 1);

  ## Calc activation values of hidden layer.
  h1 = sigmoid([ones(m, 1) X] * Theta1');
  p = [ones(m, 1) h1] * Theta2';

end
