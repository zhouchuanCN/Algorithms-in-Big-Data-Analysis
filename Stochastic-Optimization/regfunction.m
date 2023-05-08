function [loss, grad] = regfunction(x, y, w, lambda)
% x [784, batchSize]
% y [1, batchSize]
% w [784, 1]
y_hat = w'*x; % [1, batchSize]


f = 1 - tanh(y .* y_hat);  % [1 batchSize]
loss = mean(f) + lambda * w' * w;
grad = -(y .* x) ./ cosh(y .* y_hat) ./ cosh(y .* y_hat); % [784, batchSize]
grad = mean(grad, 2) + 2*lambda*w; % [784, 1]
end
