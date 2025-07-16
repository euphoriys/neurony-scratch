import numpy as np
import sys
from neural_net import NeuralNetwork

def main():
    if len(sys.argv) != 3:
        print("Usage: python test.py x1 x2")
        sys.exit(1)

    x1, x2 = float(sys.argv[1]), float(sys.argv[2])
    input_data = np.array([[x1, x2]])

    nn = NeuralNetwork([2, 4, 1])
    nn.load("model.npz")
    output = nn.predict(input_data)
    print(f"🧠 Input: {x1}, {x2} → Output: {output[0][0]:.4f}")

if __name__ == "__main__":
    main()