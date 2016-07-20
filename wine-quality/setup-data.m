## Setup winequality data workspace. 

data = dlmread('winequality-red.csv', ';', 1, 0);

## Number of features/columns. Note that this counts the output 'y' column
## which isn't a feature, but we're later going to remove it and add a column
## of 1's, so we're back to the same count.
n = size(data, 2);

## Create vector for theta (aka model weights).
theta = zeros(n,1);

mTotal = length(data);
## Split the data into training and testing sets. Simple for now.
trainingRatio = .93;

#### Setup training data ####
## Size of training set.
mTraining = uint16(mTotal * trainingRatio);

## Randomly sample data to avoid bias.
indices = randperm(mTotal, mTraining);
[XTraining, yTraining] = dataToXy(data(indices, :));

### Setup our validation data ###
## Size of hold/test set.
mTest = mTotal - mTraining;

## Grab remaining rows (unused by training set).
indices = setdiff(1:mTotal, indices);
[XTest, yTest] = dataToXy(data(indices, :));


## Safe guess on alpha (learning rate)
alpha = 1 / (max(max(XTraining)) * max(yTraining));

## Setup neural network parameters, if this method preferred.
input_layer_size  = 12;  # 12 wine attributes.
hidden_layer_size = 6;   # First guess, (input units + output units) / 2
output_layer_size = 1;   # Regression mode.

initial_Theta1 = randInitWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitWeights(hidden_layer_size, output_layer_size);

# Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

options = optimset('MaxIter', 50);
## Regularization weight.
lambda = 1;
