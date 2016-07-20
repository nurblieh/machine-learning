function [J grad] = costFunctionNN(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   output_layer_size, ...
                                   X, y, lambda)
  ## [J grad] = CostFunctionNN(nn_params, hidden_layer_size, num_labels, ...
  ## X, y, lambda) computes the cost and gradient of the neural network. The
  ## parameters for the neural network are "unrolled" into the vector
  ## nn_params and need to be converted back into weight matrices. 
  
  ## The returned parameter grad should be a "unrolled" vector of the
  ## partial derivatives of the neural network.


  ## Reshape nn_params back into Theta1 and Theta2, the weight matrices
  ## for our 2 layer neural network.
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));

  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   output_layer_size, (hidden_layer_size + 1));

  m = size(X, 1);

  ## Add a column for the bias term.
  X = [ones(m, 1) X];           # 1500x13

  ## This is converted from a previous logistic/classification implementation.
  ## Some code for that may be commented out.
  z2 = Theta1 * X';             # 6x13 * 13x1500 = 6x1500
  a2 = sigmoid(z2);
  a2 = [ones(1, columns(a2)); a2]; # 7x1500 ; Add bias unit as first row.
  z3 = Theta2 * a2;                # 1x7 * 7x1500 = 1x1500
  a3 = z3; # sigmoid(z3);

  ## No need to convert y for this regression problem.
  #y = eye(num_labels)(y,:);       # "Permutation Matrix"

  ## Cost func for logistic/classification
  ## J = sum(sum(-y' .* log(a3) - (1-y') .* log(1-a3))) / m;

  J = (1/(2*m)) * sum((a3 - y') .^ 2);

  ## Back prop stuff.
  delta_3 = a3 - y';            # 1x1500 - 1x1500 = 1x1500;
  z2 = [ones(1,m); z2];         # 7x1500
  ## Broken attempt with no sigmoids.
  ##delta_2 = (Theta2' * delta_3) .* z2; # 7x1 * 1x1500 .* 7x1500  = 7x1500
  delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2);
  delta_2 = delta_2(2:end,:);          # 6x1500

  Theta1_grad = (delta_2 * X) / m;   # 6x1500 * 1500x13 = 6x13
  Theta2_grad = (delta_3 * a2') / m; # 1x1500 * 1500x7 = 1x7

  ## Regularization
  ## Square each element of Theta1 and Theta2 and sum all elements together.
  J += (lambda/(2*m)) * (sum(sum(Theta1(:,2:end) .^ 2)) + ...
                         (sum(sum(Theta2(:,2:end) .^ 2))));

  Theta1_grad(:,2:end) += (lambda * Theta1(:,2:end)) / m;
  Theta2_grad(:,2:end) += (lambda * Theta2(:,2:end)) / m;

  ## Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
