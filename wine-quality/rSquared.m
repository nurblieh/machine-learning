## rSquared(X, y, theta)
## Measure the R-squared value of our prediction.
## https://en.wikipedia.org/wiki/Coefficient_of_determination
function R = rSquared(p, y)
  yMean = mean(y);
  R = 1 - (sum((y - p) .^ 2) / sum((y - yMean) .^ 2));
end
