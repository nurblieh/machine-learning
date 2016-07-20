function W = randInitWeights(L_in, L_out)
  ## Randomly initialize weight matrices to break symmetry of system.
  ## Args:
  ##   L_in: size of first layer.
  ##   L_out: size of second layer.
  epsilon = 0.12;
  W = rand(L_out, 1+L_in) * 2 * epsilon - epsilon;
end
