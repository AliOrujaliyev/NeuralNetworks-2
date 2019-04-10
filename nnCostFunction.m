function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
<<<<<<< HEAD
=======
K = num_labels;
>>>>>>> 4ee63c8c0b53d90afd440d55b15edd94dd09fe8e
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


<<<<<<< HEAD
X = [ones(m, 1), X];
h = sigmoid(X*Theta1');
h = [ones(m, 1), h];
h2 = sigmoid(h*Theta2');



y2 = zeros(num_labels, size(y));

for i=1:m
    y2(y(i), i) = 1;
end

theta1 = Theta1(:,2:end);
theta2 = Theta2(:,2:end);
J = (1/m) * sum(sum(-y2 .* log(h2') - (1-y2) .* log(1-h2'))) + (lambda/(2*m)) * ((sum(sum(theta1.^2))) + (sum(sum(theta2.^2))));

=======
disp(size(Theta1));
disp(size(Theta2));
disp(size(K));
disp(size(input_layer_size));
disp(size(hidden_layer_size));



h = sigmoid(X*Theta1);
for i = i:m

end 
>>>>>>> 4ee63c8c0b53d90afd440d55b15edd94dd09fe8e





%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
<<<<<<< HEAD

delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));
y2 = y2';

for t=1:m
    a1t = X(t,:)';
    a2t = h(t,:)';
    ht = h2(t,:)';
    yt = y2(t,:)';

    dlt3 = ht - yt;
    z2t = Theta1 * a1t;
    z2t = [1; z2t];
    dlt2 = Theta2' * dlt3 .* sigmoidGradient(z2t);

    delta1 = delta1 + dlt2(2:end) * (a1t)';
    delta2 = delta2 + dlt3 * (a2t)';


    
end

Theta1_grad = (1/m) * delta1;
Theta2_grad = (1/m) * delta2;





=======
>>>>>>> 4ee63c8c0b53d90afd440d55b15edd94dd09fe8e
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


<<<<<<< HEAD
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);
=======


>>>>>>> 4ee63c8c0b53d90afd440d55b15edd94dd09fe8e















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
