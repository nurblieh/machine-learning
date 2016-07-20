## Run our neural network.
## Uses the training data to build the model.
## Outputs prediction results over training and test.

## Function handle to pass to linear solver.
f = @(p) costFunctionNN(p, ...
                        input_layer_size, hidden_layer_size, output_layer_size, ...
                        XTraining, yTraining, lambda);

# Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf("Running neural network for %d iterations... ", options.MaxIter);
fflush(1);
# [nn_params, cost] = fmincg(f, initial_nn_params, options);
[nn_params, cost, exit_flag] = fminunc(f, initial_nn_params, options);
fprintf("\n");

## Reshape the unrolled theta vector into two matrices.
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 output_layer_size, (hidden_layer_size + 1));

## Generate our predictions.
pTraining = predictNN(Theta1, Theta2, XTraining);
pTest = predictNN(Theta1, Theta2, XTest);

## Display stats.
printf("Cost (training set): %0.4f\n", squaredErrors(pTraining, yTraining));
printf("Cost (test set): %0.4f\n", squaredErrors(pTest, yTest));

printf("R-Squared (training set): %0.4f\n", rSquared(pTraining, yTraining));
printf("R-Squared (test set): %0.4f\n", rSquared(pTest, yTest));
