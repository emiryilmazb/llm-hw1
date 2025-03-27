import numpy as np


class NeuralNetwork:
    def __init__(self, learning_rate=0.1):
        # Initialize weights and biases with the values from our solution
        # Input to hidden layer weights
        self.w11 = 0.1   # x1 to h1
        self.w21 = -0.2  # x2 to h1
        self.w31 = 0.3   # x3 to h1
        self.w12 = -0.1  # x1 to h2
        self.w22 = 0.2   # x2 to h2
        self.w32 = -0.3  # x3 to h2

        # Hidden to output layer weights
        self.v11 = 0.4   # h1 to o1
        self.v21 = -0.4  # h2 to o1
        self.v12 = -0.5  # h1 to o2
        self.v22 = 0.5   # h2 to o2

        # Bias values
        self.b1 = 0.1    # bias for h1
        self.b2 = -0.1   # bias for h2
        self.b3 = 0.2    # bias for o1
        self.b4 = -0.2   # bias for o2

        self.learning_rate = learning_rate

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)

    def forward_propagation(self, inputs):
        """Forward pass through the network"""
        x1, x2, x3 = inputs

        # Calculate hidden layer activations
        net_h1 = self.w11 * x1 + self.w21 * x2 + self.w31 * x3 + self.b1
        self.h1 = self.sigmoid(net_h1)

        net_h2 = self.w12 * x1 + self.w22 * x2 + self.w32 * x3 + self.b2
        self.h2 = self.sigmoid(net_h2)

        # Calculate output layer activations
        net_o1 = self.v11 * self.h1 + self.v21 * self.h2 + self.b3
        self.o1 = self.sigmoid(net_o1)

        net_o2 = self.v12 * self.h1 + self.v22 * self.h2 + self.b4
        self.o2 = self.sigmoid(net_o2)

        return [self.o1, self.o2]

    def compute_error(self, targets):
        """Calculate Mean Squared Error"""
        t1, t2 = targets
        error = 0.5 * ((self.o1 - t1)**2 + (self.o2 - t2)**2)
        return error

    def backward_propagation(self, inputs, targets):
        """Backpropagation algorithm"""
        x1, x2, x3 = inputs
        t1, t2 = targets

        # Calculate output layer error derivatives
        dE_do1 = self.o1 - t1
        dE_do2 = self.o2 - t2

        # Calculate output layer deltas
        delta_o1 = dE_do1 * self.sigmoid_derivative(self.o1)
        delta_o2 = dE_do2 * self.sigmoid_derivative(self.o2)

        # Calculate hidden layer deltas
        delta_h1 = (delta_o1 * self.v11 + delta_o2 * self.v12) * \
            self.sigmoid_derivative(self.h1)
        delta_h2 = (delta_o1 * self.v21 + delta_o2 * self.v22) * \
            self.sigmoid_derivative(self.h2)

        # Update weights and biases
        # Hidden to output layer
        self.v11 -= self.learning_rate * delta_o1 * self.h1
        self.v21 -= self.learning_rate * delta_o1 * self.h2
        self.v12 -= self.learning_rate * delta_o2 * self.h1
        self.v22 -= self.learning_rate * delta_o2 * self.h2

        # Input to hidden layer
        self.w11 -= self.learning_rate * delta_h1 * x1
        self.w21 -= self.learning_rate * delta_h1 * x2
        self.w31 -= self.learning_rate * delta_h1 * x3
        self.w12 -= self.learning_rate * delta_h2 * x1
        self.w22 -= self.learning_rate * delta_h2 * x2
        self.w32 -= self.learning_rate * delta_h2 * x3

        # Update biases
        self.b1 -= self.learning_rate * delta_h1
        self.b2 -= self.learning_rate * delta_h2
        self.b3 -= self.learning_rate * delta_o1
        self.b4 -= self.learning_rate * delta_o2

        return delta_h1, delta_h2, delta_o1, delta_o2

    def train(self, inputs, targets):
        """Single training step"""
        # Forward pass
        outputs = self.forward_propagation(inputs)

        # Calculate error
        error = self.compute_error(targets)

        # Backward pass
        deltas = self.backward_propagation(inputs, targets)

        return outputs, error, deltas

    def print_weights(self):
        """Print all weights and biases"""
        print("Input to Hidden Layer Weights:")
        print(
            f"w11 = {self.w11:.4f}, w21 = {self.w21:.4f}, w31 = {self.w31:.4f}")
        print(
            f"w12 = {self.w12:.4f}, w22 = {self.w22:.4f}, w32 = {self.w32:.4f}")
        print("\nHidden to Output Layer Weights:")
        print(f"v11 = {self.v11:.4f}, v21 = {self.v21:.4f}")
        print(f"v12 = {self.v12:.4f}, v22 = {self.v22:.4f}")
        print("\nBiases:")
        print(
            f"b1 = {self.b1:.4f}, b2 = {self.b2:.4f}, b3 = {self.b3:.4f}, b4 = {self.b4:.4f}")


# Main execution
if __name__ == "__main__":
    # Initialize neural network
    nn = NeuralNetwork(learning_rate=0.1)

    # Input and target values from the problem
    inputs = [2, 3, 7]
    targets = [0.1, 0.05]

    print("Initial weights and biases:")
    nn.print_weights()

    # Forward propagation
    outputs = nn.forward_propagation(inputs)
    print("\nForward Propagation Results:")
    print(f"h1 = {nn.h1:.4f}, h2 = {nn.h2:.4f}")
    print(f"o1 = {outputs[0]:.4f}, o2 = {outputs[1]:.4f}")

    # Calculate error
    error = nn.compute_error(targets)
    print(f"\nInitial error (MSE): {error:.4f}")

    # Perform backpropagation and update weights
    deltas = nn.backward_propagation(inputs, targets)
    print("\nBackpropagation Deltas:")
    print(f"delta_h1 = {deltas[0]:.4f}, delta_h2 = {deltas[1]:.4f}")
    print(f"delta_o1 = {deltas[2]:.4f}, delta_o2 = {deltas[3]:.4f}")

    # Display updated weights
    print("\nUpdated weights and biases after one training step:")
    nn.print_weights()

    # Verify with a second forward pass
    new_outputs = nn.forward_propagation(inputs)
    new_error = nn.compute_error(targets)
    print(
        f"\nNew outputs: o1 = {new_outputs[0]:.4f}, o2 = {new_outputs[1]:.4f}")
    print(f"New error (MSE): {new_error:.4f}")

    # Check if error decreased
    print(f"Error reduction: {error - new_error:.6f}")
