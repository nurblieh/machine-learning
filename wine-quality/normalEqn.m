## NORMALEQN Computes the closed-form solution to linear regression 
## NORMALEQN(X,y) computes the closed-form solution to linear 
## regression using the normal equations.
## Note that X should be invertible.
## https://en.wikipedia.org/wiki/Invertible_matrix
function [theta] = normalEqn(X, y)  
  theta = pinv(X' * X) * X' * y;
end


