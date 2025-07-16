# Neurony-Scratch

A simple Python implementation of a Multi-Layer Perceptron (MLP) neural network from scratch.  
Designed to simulate logical gates like XOR and easily extendable.

## Features

- Customizable layer sizes
- Sigmoid activation function
- Training using backpropagation
- Save and load model parameters
- Command line interface for training and testing

## Files

- `neural_net.py`: Neural network implementation
- `train.py`: Train the neural network with custom output values (e.g. XOR outputs)
- `test.py`: Test the trained model with input values
- `model.npz`: Saved model parameters (ignored by Git by default)
- `requirements.txt`: Python dependencies

## Usage

### Training

Train the model by providing 4 output values (for inputs `[0,0], [0,1], [1,0], [1,1]`):

```bash
python train.py 0 1 1 0
```

This will save the model to `model.npz`

## Testing

Test the model with two input values:

```python
python test.py 1 0
```

This script will output the model's prediction.

# Requirements
- Python 3.x
- numpy

Install dependencies with:

```python
pip install -r requirements.txt
```

### Created by euphoriys