import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        # Initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            self.weights.append(np.random.uniform(-1, 1, (input_size, output_size)))
            self.biases.append(np.zeros((1, output_size)))

    def forward(self, x):
        self.z_values = []
        self.activations = [x]
        a = x
        for W, b in zip(self.weights, self.biases):
            z = np.dot(a, W) + b
            a = sigmoid(z)
            self.z_values.append(z)
            self.activations.append(a)
        return a

    def backward(self, y_true):
        deltas = []
        error = y_true - self.activations[-1]
        delta = error * sigmoid_derivative(self.activations[-1])
        deltas.append(delta)

        # Go backward through the layers
        for l in range(len(self.layer_sizes) - 2, 0, -1):
            delta = deltas[-1].dot(self.weights[l].T) * sigmoid_derivative(self.activations[l])
            deltas.append(delta)

        deltas.reverse()

        # Update weights and biases
        for i in range(len(self.weights)):
            a = self.activations[i]
            d = deltas[i]
            self.weights[i] += a.T.dot(d) * self.learning_rate
            self.biases[i] += np.sum(d, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(y)
            if epoch % 1000 == 0:
                loss = np.mean((y - output) ** 2)
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

    def predict(self, x):
        return self.forward(x)

    def save(self, path="model.npz"):
        # Convert weights and biases lists to np arrays with dtype=object
        weights_obj = [w for w in self.weights]
        biases_obj = [b.reshape(-1) for b in self.biases]  # flatten biases to 1D arrays
        np.savez(path,
                 weights=np.array(weights_obj, dtype=object),
                 biases=np.array(biases_obj, dtype=object),
                 layer_sizes=np.array(self.layer_sizes))

    def load(self, path="model.npz"):
        data = np.load(path, allow_pickle=True)
        self.weights = list(data['weights'])
        self.biases = [b.reshape(1, -1) for b in data['biases']]
        self.layer_sizes = list(data['layer_sizes'])