## Function to convert the dlmread() results into something usable.
## Ret:
##   X: Feature matrix. Column 1 will be all 1's.
##   y: Output/label vector. What we want to predict.
function [X, y] = dataToXy(data)
  [m, n] = size(data);
  # Add col of ones and strip output values.
  X = [ones(m, 1), data(:, 1:n-1)];
  y = data(:, n);
end
