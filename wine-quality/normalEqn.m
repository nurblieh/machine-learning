## normalEqn(X,y) computes the closed-form solution to linear regression using
## the normal equations.
##
## If you're bored,
## http://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression/
##
## Ret:
##   theta: A vector of weights which should minimize our loss!
function [theta] = normalEqn(X, y)  
  theta = pinv(X' * X) * X' * y;
end


