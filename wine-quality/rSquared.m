## rSquared(X, y, theta)
## Measure the R-squared value of our prediction.
## https://en.wikipedia.org/wiki/Coefficient_of_determination
function R = rSquared(X, y, theta)
  yMean = mean(y);
  yPrediction = X * theta;
  R = 1 - (sum((y - yPrediction) .^ 2) / sum((y - yMean) .^ 2));
end
