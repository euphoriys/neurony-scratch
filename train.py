import numpy as np
import sys
from neural_net import NeuralNetwork

def main():
    if len(sys.argv) != 5:
        print("Usage: python train.py y00 y01 y10 y11")
        print("Example (XOR): python train.py 0 1 1 0")
        sys.exit(1)

    # Convert arguments to a NumPy array
    try:
        y_values = [float(arg) for arg in sys.argv[1:]]
    except ValueError:
        print("All outputs must be numeric (0 or 1).")
        sys.exit(1)

    y = np.array(y_values).reshape(-1, 1)

    # Input combinations for logic gates
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # Create and train the network
    nn = NeuralNetwork([2, 4, 1], learning_rate=0.2)
    nn.train(X, y, epochs=20000)

    # Save the model
    nn.save("model.npz")
    print("✅ Model saved as 'model.npz'")

if __name__ == "__main__":
    main()
