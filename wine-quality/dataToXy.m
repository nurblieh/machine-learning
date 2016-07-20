## Function to convert the dlmread() results into something usable.
## Ret:
##   X: Feature matrix.
##   y: Output/label vector.
function [X, y] = dataToXy(data)
  [m, n] = size(data);
  X = data(:, 1:n-1);
  y = data(:, n);
end
