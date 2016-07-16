## Setup winequality data workspace. 

data = dlmread('winequality-red.csv', ';', 1, 0);

## Number of features/columns. Note that this counts the output 'y' column
## which isn't a feature, but we're later going to remove it and add a column
## of 1's, so we're back to the same count.
n = size(data, 2);

## Create vector for theta (aka model weights).
theta = zeros(n,1);

mTotal = length(data);
## Use 90% of our data for training; 10% for validation.
trainingRatio = .9;

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


