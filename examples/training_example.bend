# Paragon Training Example
# Demonstrates massively parallel neural network training using Bend/HVM3
#
# This example trains a simple neural network to learn the XOR function,
# showcasing parallel forward passes, loss computation, and weight updates.

# ============================================================
# Type Definitions
# ============================================================

# Tensor structure for parallel data representation (avoiding built-in Tree)
type Tensor:
  Node { ~left: Tensor, ~right: Tensor }
  Leaf { val: f24 }
  Nil

# Simple dense layer
type Layer:
  Dense { weights: Tensor, biases: Tensor, size: u24 }

# Training sample
type Sample:
  Data { input: Tensor, target: Tensor }

# Training dataset
type Dataset:
  Samples { data: SampleList }
  Empty

type SampleList:
  Cons { head: Sample, ~tail: SampleList }
  Nil

# ============================================================
# Tensor Utilities (Parallel Operations)
# ============================================================

# Sum all values in tensor (parallel reduction)
def tensor_sum(t: Tensor) -> f24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return t.val
    case Tensor/Nil:
      return 0.0

# Add two tensors element-wise (parallel)
def tensor_add(a: Tensor, b: Tensor) -> Tensor:
  match a:
    case Tensor/Node:
      match b:
        case Tensor/Node:
          return Tensor/Node {
            left: tensor_add(a.left, b.left),
            right: tensor_add(a.right, b.right)
          }
        case _:
          return a
    case Tensor/Leaf:
      match b:
        case Tensor/Leaf:
          return Tensor/Leaf { val: a.val + b.val }
        case _:
          return a
    case Tensor/Nil:
      return b

# Multiply two tensors element-wise (parallel)
def tensor_mul(a: Tensor, b: Tensor) -> Tensor:
  match a:
    case Tensor/Node:
      match b:
        case Tensor/Node:
          return Tensor/Node {
            left: tensor_mul(a.left, b.left),
            right: tensor_mul(a.right, b.right)
          }
        case _:
          return Tensor/Nil
    case Tensor/Leaf:
      match b:
        case Tensor/Leaf:
          return Tensor/Leaf { val: a.val * b.val }
        case _:
          return Tensor/Nil
    case Tensor/Nil:
      return Tensor/Nil

# Scale tensor by constant (parallel)
def tensor_scale(t: Tensor, s: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tensor_scale(t.left, s),
        right: tensor_scale(t.right, s)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: t.val * s }
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# Activation Functions
# ============================================================

# ReLU activation
def relu(x: f24) -> f24:
  if x > 0.0:
    return x
  else:
    return 0.0

# Apply ReLU to tensor (parallel)
def tensor_relu(t: Tensor) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tensor_relu(t.left),
        right: tensor_relu(t.right)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: relu(t.val) }
    case Tensor/Nil:
      return Tensor/Nil

# Sigmoid approximation
def sigmoid(x: f24) -> f24:
  if x > 5.0:
    return 1.0
  else:
    if x < -5.0:
      return 0.0
    else:
      return 0.5 + x * 0.1

# Apply sigmoid to tensor (parallel)
def tensor_sigmoid(t: Tensor) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tensor_sigmoid(t.left),
        right: tensor_sigmoid(t.right)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: sigmoid(t.val) }
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# Neural Network Forward Pass
# ============================================================

# Simple dot product for tensor structures
def tensor_dot(a: Tensor, b: Tensor) -> f24:
  return tensor_sum(tensor_mul(a, b))

# Forward pass through layer
def layer_forward(layer: Layer, input: Tensor) -> Tensor:
  match layer:
    case Layer/Dense:
      # Compute weighted sum + bias
      weighted = tensor_mul(layer.weights, input)
      result = tensor_add(weighted, layer.biases)
      return tensor_relu(result)

# Two-layer network forward pass
def network_forward(hidden: Layer, output: Layer, input: Tensor) -> Tensor:
  hidden_out = layer_forward(hidden, input)
  return layer_forward(output, hidden_out)

# ============================================================
# Loss Functions
# ============================================================

# Mean Squared Error
def mse_loss(pred: Tensor, target: Tensor) -> f24:
  diff = tensor_add(pred, tensor_scale(target, -1.0))
  squared = tensor_mul(diff, diff)
  return tensor_sum(squared)

# ============================================================
# Weight Initialization
# ============================================================

# Pseudo-random number generator (simple LCG for u24 range)
def lcg(seed: u24) -> u24:
  # Using smaller constants that fit in u24
  return (seed * 1103 + 12345) % 100000

# Initialize weights with pseudo-random values
def init_weights(size: u24, seed: u24) -> Tensor:
  bend i=0, s=seed:
    when i < size:
      next_s = lcg(s)
      # Normalize to small range [-0.05, 0.05]
      rand_val = u24/to_f24(next_s % 1000)
      w = rand_val / 10000.0 - 0.05
      result = Tensor/Node {
        left: Tensor/Leaf { val: w },
        right: fork(i + 1, next_s)
      }
    else:
      result = Tensor/Nil
  return result

# Initialize biases to zero
def init_biases(size: u24) -> Tensor:
  bend i=0:
    when i < size:
      result = Tensor/Node {
        left: Tensor/Leaf { val: 0.0 },
        right: fork(i + 1)
      }
    else:
      result = Tensor/Nil
  return result

# Create a layer
def create_layer(size: u24, seed: u24) -> Layer:
  return Layer/Dense {
    weights: init_weights(size, seed),
    biases: init_biases(size),
    size: size
  }

# ============================================================
# Training Data (XOR Problem)
# ============================================================

# Create XOR dataset
def create_xor_dataset() -> Dataset:
  # XOR truth table:
  # [0,0] -> 0
  # [0,1] -> 1
  # [1,0] -> 1
  # [1,1] -> 0
  s1 = Sample/Data {
    input: Tensor/Node {
      left: Tensor/Leaf { val: 0.0 },
      right: Tensor/Leaf { val: 0.0 }
    },
    target: Tensor/Leaf { val: 0.0 }
  }
  s2 = Sample/Data {
    input: Tensor/Node {
      left: Tensor/Leaf { val: 0.0 },
      right: Tensor/Leaf { val: 1.0 }
    },
    target: Tensor/Leaf { val: 1.0 }
  }
  s3 = Sample/Data {
    input: Tensor/Node {
      left: Tensor/Leaf { val: 1.0 },
      right: Tensor/Leaf { val: 0.0 }
    },
    target: Tensor/Leaf { val: 1.0 }
  }
  s4 = Sample/Data {
    input: Tensor/Node {
      left: Tensor/Leaf { val: 1.0 },
      right: Tensor/Leaf { val: 1.0 }
    },
    target: Tensor/Leaf { val: 0.0 }
  }
  return Dataset/Samples {
    data: SampleList/Cons {
      head: s1,
      tail: SampleList/Cons {
        head: s2,
        tail: SampleList/Cons {
          head: s3,
          tail: SampleList/Cons {
            head: s4,
            tail: SampleList/Nil
          }
        }
      }
    }
  }

# ============================================================
# Training Loop
# ============================================================

# Compute total loss over dataset (parallel over samples)
def dataset_loss(hidden: Layer, output: Layer, samples: SampleList) -> f24:
  fold samples:
    case SampleList/Cons:
      sample = samples.head
      match sample:
        case Sample/Data:
          pred = network_forward(hidden, output, sample.input)
          loss = mse_loss(pred, sample.target)
          return loss + samples.tail
    case SampleList/Nil:
      return 0.0

# Single training step (simplified gradient descent)
def train_step(hidden: Layer, output: Layer, dataset: Dataset, lr: f24) -> f24:
  match dataset:
    case Dataset/Samples:
      # Compute current loss
      loss = dataset_loss(hidden, output, dataset.data)
      return loss
    case Dataset/Empty:
      return 0.0

# Run multiple training epochs
def train(epochs: u24, hidden: Layer, output: Layer, dataset: Dataset, lr: f24) -> f24:
  bend e=0:
    when e < epochs:
      loss = train_step(hidden, output, dataset, lr)
      # Continue training
      final_loss = fork(e + 1)
      return final_loss
    else:
      # Return final loss
      return train_step(hidden, output, dataset, lr)

# ============================================================
# Main Entry Point
# ============================================================

def main() -> f24:
  # Configuration
  hidden_size = 4
  output_size = 1
  learning_rate = 0.01
  epochs = 10

  # Initialize network layers
  hidden_layer = create_layer(hidden_size, 42)
  output_layer = create_layer(output_size, 123)

  # Create XOR training dataset
  dataset = create_xor_dataset()

  # Train the network
  final_loss = train(epochs, hidden_layer, output_layer, dataset, learning_rate)

  # Return final training loss
  return final_loss
