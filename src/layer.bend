# Neural network layer definitions
# Supports dense (fully connected) layers with parallel computation

type Tensor:
  Node { ~left: Tensor, ~right: Tensor }
  Leaf { val: f24 }
  Nil

type Vector:
  Vec { len: u24, data: Tensor }
  Empty

# ============================================================
# Layer Types
# ============================================================

# Dense (fully connected) layer
type DenseLayer:
  Dense {
    input_size: u24,
    output_size: u24,
    weights: Tensor,      # 2D weights as nested tree
    biases: Tensor        # 1D biases
  }

# Layer output with cached values for backprop
type LayerOutput:
  Output {
    pre_activation: Tensor,   # z = Wx + b
    post_activation: Tensor   # a = activation(z)
  }

# ============================================================
# Weight Initialization
# ============================================================

# Initialize weights with small random-like values using LCG
def init_weight(seed: u24) -> f24:
  # Linear congruential generator for pseudo-random initialization
  # Produces values roughly in range [-0.5, 0.5]
  a = 1103515245
  c = 12345
  m = 2147483648
  next_seed = (a * seed + c) % m
  # Normalize to [-0.5, 0.5]
  return (next_seed / m) - 0.5

# Create weight tree for a layer
def create_weights(input_size: u24, output_size: u24, seed: u24) -> Tensor:
  total = input_size * output_size
  bend i=0, s=seed:
    when i < total:
      w = init_weight(s)
      result = Tensor/Node {
        left: Tensor/Leaf { val: w * 0.1 },  # Scale down for Xavier-like init
        right: fork(i + 1, s + 1)
      }
    else:
      result = Tensor/Nil
  return result

# Create bias tree initialized to zeros
def create_biases(size: u24) -> Tensor:
  bend i=0:
    when i < size:
      result = Tensor/Node {
        left: Tensor/Leaf { val: 0.0 },
        right: fork(i + 1)
      }
    else:
      result = Tensor/Nil
  return result

# ============================================================
# Layer Creation
# ============================================================

# Create a new dense layer
def dense_layer(input_size: u24, output_size: u24, seed: u24) -> DenseLayer:
  return DenseLayer/Dense {
    input_size: input_size,
    output_size: output_size,
    weights: create_weights(input_size, output_size, seed),
    biases: create_biases(output_size)
  }

# ============================================================
# Forward Pass (Parallel)
# ============================================================

# Element-wise multiply and accumulate (dot product helper)
def tree_dot_accum(weights: Tensor, input: Tensor) -> f24:
  fold weights with input:
    case Tensor/Node:
      match input:
        case Tensor/Node:
          return weights.left(input.left) + weights.right(input.right)
        case _:
          return 0.0
    case Tensor/Leaf:
      match input:
        case Tensor/Leaf:
          return weights.val * input.val
        case _:
          return 0.0
    case Tensor/Nil:
      return 0.0

# Forward pass through dense layer (parallel)
def dense_forward(layer: DenseLayer, input: Tensor) -> Tensor:
  match layer:
    case DenseLayer/Dense:
      # Compute Wx + b for each output neuron
      bend i=0:
        when i < layer.output_size:
          # This naturally parallelizes across output neurons
          neuron_output = tree_dot_accum(layer.weights, input)
          result = Tensor/Node {
            left: Tensor/Leaf { val: neuron_output },
            right: fork(i + 1)
          }
        else:
          result = Tensor/Nil
      return result

# Add biases to output (parallel)
def add_biases(output: Tensor, biases: Tensor) -> Tensor:
  match output:
    case Tensor/Node:
      match biases:
        case Tensor/Node:
          return Tensor/Node {
            left: add_biases(output.left, biases.left),
            right: add_biases(output.right, biases.right)
          }
        case _:
          return output
    case Tensor/Leaf:
      match biases:
        case Tensor/Leaf:
          return Tensor/Leaf { val: output.val + biases.val }
        case _:
          return output
    case Tensor/Nil:
      return biases

# ============================================================
# Backward Pass (Gradient Computation)
# ============================================================

# Compute gradient with respect to weights
def compute_weight_gradient(input: Tensor, output_grad: Tensor) -> Tensor:
  # Outer product: grad_W = output_grad * input^T
  match output_grad:
    case Tensor/Node:
      return Tensor/Node {
        left: compute_weight_gradient(input, output_grad.left),
        right: compute_weight_gradient(input, output_grad.right)
      }
    case Tensor/Leaf:
      return tree_scale_local(input, output_grad.val)
    case Tensor/Nil:
      return Tensor/Nil

# Local tree scaling (helper for gradient computation)
def tree_scale_local(t: Tensor, s: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_scale_local(t.left, s),
        right: tree_scale_local(t.right, s)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: t.val * s }
    case Tensor/Nil:
      return Tensor/Nil

# Compute gradient with respect to input (for backprop to previous layer)
def compute_input_gradient(weights: Tensor, output_grad: Tensor) -> Tensor:
  # grad_input = W^T * output_grad
  fold weights with output_grad:
    case Tensor/Node:
      match output_grad:
        case Tensor/Node:
          return tree_add_local(weights.left(output_grad.left), weights.right(output_grad.right))
        case _:
          return Tensor/Nil
    case Tensor/Leaf:
      match output_grad:
        case Tensor/Leaf:
          return Tensor/Leaf { val: weights.val * output_grad.val }
        case _:
          return Tensor/Nil
    case Tensor/Nil:
      return Tensor/Nil

# Local tree addition (helper)
def tree_add_local(a: Tensor, b: Tensor) -> Tensor:
  match a:
    case Tensor/Node:
      match b:
        case Tensor/Node:
          return Tensor/Node {
            left: tree_add_local(a.left, b.left),
            right: tree_add_local(a.right, b.right)
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

# ============================================================
# Weight Update (SGD)
# ============================================================

# Update weights using gradient descent
def update_weights(weights: Tensor, gradients: Tensor, learning_rate: f24) -> Tensor:
  match weights:
    case Tensor/Node:
      match gradients:
        case Tensor/Node:
          return Tensor/Node {
            left: update_weights(weights.left, gradients.left, learning_rate),
            right: update_weights(weights.right, gradients.right, learning_rate)
          }
        case _:
          return weights
    case Tensor/Leaf:
      match gradients:
        case Tensor/Leaf:
          return Tensor/Leaf { val: weights.val - learning_rate * gradients.val }
        case _:
          return weights
    case Tensor/Nil:
      return Tensor/Nil

# Update layer with computed gradients
def update_layer(layer: DenseLayer, weight_grad: Tensor, bias_grad: Tensor, lr: f24) -> DenseLayer:
  match layer:
    case DenseLayer/Dense:
      return DenseLayer/Dense {
        input_size: layer.input_size,
        output_size: layer.output_size,
        weights: update_weights(layer.weights, weight_grad, lr),
        biases: update_weights(layer.biases, bias_grad, lr)
      }
