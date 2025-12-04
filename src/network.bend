# Neural Network definitions
# Combines layers into a complete trainable network

type Tensor:
  Node { ~left: Tensor, ~right: Tensor }
  Leaf { val: f24 }
  Nil

# Dense layer type
type DenseLayer:
  Dense {
    input_size: u24,
    output_size: u24,
    weights: Tensor,
    biases: Tensor
  }

# Activation type enum
type Activation:
  ReLU
  Sigmoid
  Tanh
  LeakyReLU
  Linear
  Softmax

# Layer with activation
type Layer:
  LayerWithAct {
    dense: DenseLayer,
    activation: Activation
  }

# Neural network as a list of layers
type Network:
  Sequential { layers: LayerList }
  Empty

type LayerList:
  Cons { head: Layer, ~tail: LayerList }
  Nil

# Training configuration
type TrainConfig:
  Config {
    learning_rate: f24,
    epochs: u24,
    batch_size: u24
  }

# ============================================================
# Network Creation
# ============================================================

# Create an empty network
def network_new() -> Network:
  return Network/Sequential { layers: LayerList/Nil }

# Add a layer to the network
def network_add_layer(net: Network, layer: Layer) -> Network:
  match net:
    case Network/Sequential:
      return Network/Sequential {
        layers: layer_list_append(net.layers, layer)
      }
    case Network/Empty:
      return Network/Sequential {
        layers: LayerList/Cons { head: layer, tail: LayerList/Nil }
      }

# Append layer to list
def layer_list_append(list: LayerList, layer: Layer) -> LayerList:
  match list:
    case LayerList/Cons:
      return LayerList/Cons {
        head: list.head,
        tail: layer_list_append(list.tail, layer)
      }
    case LayerList/Nil:
      return LayerList/Cons { head: layer, tail: LayerList/Nil }

# ============================================================
# Activation Application
# ============================================================

# Apply activation function to tree
def apply_activation(t: Tensor, act: Activation) -> Tensor:
  match act:
    case Activation/ReLU:
      return tree_relu_net(t)
    case Activation/Sigmoid:
      return tree_sigmoid_net(t)
    case Activation/Tanh:
      return tree_tanh_net(t)
    case Activation/LeakyReLU:
      return tree_leaky_relu_net(t)
    case Activation/Linear:
      return t
    case Activation/Softmax:
      return tree_softmax_net(t)

# Softmax on tree (numerically stable)
def tree_softmax_net(t: Tensor) -> Tensor:
  max_val = tree_max_net(t)
  exp_shifted = tree_exp_shifted_net(t, max_val)
  sum_exp = tree_sum_net(exp_shifted)
  return tree_divide_net(exp_shifted, sum_exp)

# Find max value in tree
def tree_max_net(t: Tensor) -> f24:
  fold t:
    case Tensor/Node:
      left_max = t.left
      right_max = t.right
      if left_max > right_max:
        return left_max
      else:
        return right_max
    case Tensor/Leaf:
      return t.val
    case Tensor/Nil:
      return -999999.0

# Exponential approximation
def exp_approx_net(x: f24) -> f24:
  if x > 10.0:
    return 22026.0
  else:
    if x < -10.0:
      return 0.00005
    else:
      x2 = x * x
      x3 = x2 * x
      x4 = x3 * x
      x5 = x4 * x
      return 1.0 + x + x2 * 0.5 + x3 * 0.166667 + x4 * 0.041667 + x5 * 0.008333

# Compute exp(x - max) for each element
def tree_exp_shifted_net(t: Tensor, max_val: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_exp_shifted_net(t.left, max_val),
        right: tree_exp_shifted_net(t.right, max_val)
      }
    case Tensor/Leaf:
      x = t.val - max_val
      return Tensor/Leaf { val: exp_approx_net(x) }
    case Tensor/Nil:
      return Tensor/Nil

# Divide each element by divisor
def tree_divide_net(t: Tensor, divisor: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_divide_net(t.left, divisor),
        right: tree_divide_net(t.right, divisor)
      }
    case Tensor/Leaf:
      if divisor > 0.0001:
        return Tensor/Leaf { val: t.val / divisor }
      else:
        return Tensor/Leaf { val: 0.0 }
    case Tensor/Nil:
      return Tensor/Nil

# ReLU on tree
def tree_relu_net(t: Tensor) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_relu_net(t.left),
        right: tree_relu_net(t.right)
      }
    case Tensor/Leaf:
      if t.val > 0.0:
        return Tensor/Leaf { val: t.val }
      else:
        return Tensor/Leaf { val: 0.0 }
    case Tensor/Nil:
      return Tensor/Nil

# Sigmoid on tree
def tree_sigmoid_net(t: Tensor) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_sigmoid_net(t.left),
        right: tree_sigmoid_net(t.right)
      }
    case Tensor/Leaf:
      x = t.val
      if x > 5.0:
        return Tensor/Leaf { val: 1.0 }
      else:
        if x < -5.0:
          return Tensor/Leaf { val: 0.0 }
        else:
          return Tensor/Leaf { val: 0.5 + x * 0.1 }
    case Tensor/Nil:
      return Tensor/Nil

# Tanh on tree
def tree_tanh_net(t: Tensor) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_tanh_net(t.left),
        right: tree_tanh_net(t.right)
      }
    case Tensor/Leaf:
      x = t.val
      if x > 3.0:
        return Tensor/Leaf { val: 1.0 }
      else:
        if x < -3.0:
          return Tensor/Leaf { val: -1.0 }
        else:
          x2 = x * x
          return Tensor/Leaf { val: x * (27.0 + x2) / (27.0 + 9.0 * x2) }
    case Tensor/Nil:
      return Tensor/Nil

# Leaky ReLU on tree
def tree_leaky_relu_net(t: Tensor) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_leaky_relu_net(t.left),
        right: tree_leaky_relu_net(t.right)
      }
    case Tensor/Leaf:
      if t.val > 0.0:
        return Tensor/Leaf { val: t.val }
      else:
        return Tensor/Leaf { val: 0.01 * t.val }
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# Forward Pass
# ============================================================

# Forward pass through single layer
def layer_forward(layer: Layer, input: Tensor) -> Tensor:
  match layer:
    case Layer/LayerWithAct:
      # Linear transformation
      linear_out = dense_forward_net(layer.dense, input)
      # Apply activation
      return apply_activation(linear_out, layer.activation)

# Dense forward pass
def dense_forward_net(dense: DenseLayer, input: Tensor) -> Tensor:
  match dense:
    case DenseLayer/Dense:
      # Simplified: element-wise computation with biases
      weighted = tree_weighted_sum(dense.weights, input)
      return tree_add_net(weighted, dense.biases)

# Weighted sum (simplified matrix-vector product)
def tree_weighted_sum(weights: Tensor, input: Tensor) -> Tensor:
  match weights:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_weighted_sum(weights.left, input),
        right: tree_weighted_sum(weights.right, input)
      }
    case Tensor/Leaf:
      match input:
        case Tensor/Leaf:
          return Tensor/Leaf { val: weights.val * input.val }
        case _:
          return Tensor/Leaf { val: 0.0 }
    case Tensor/Nil:
      return Tensor/Nil

# Add two trees
def tree_add_net(a: Tensor, b: Tensor) -> Tensor:
  match a:
    case Tensor/Node:
      match b:
        case Tensor/Node:
          return Tensor/Node {
            left: tree_add_net(a.left, b.left),
            right: tree_add_net(a.right, b.right)
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

# Forward pass through entire network
def network_forward(net: Network, input: Tensor) -> Tensor:
  match net:
    case Network/Sequential:
      return layers_forward(net.layers, input)
    case Network/Empty:
      return input

# Forward through layer list
def layers_forward(layers: LayerList, input: Tensor) -> Tensor:
  match layers:
    case LayerList/Cons:
      output = layer_forward(layers.head, input)
      return layers_forward(layers.tail, output)
    case LayerList/Nil:
      return input

# ============================================================
# Training Utilities
# ============================================================

# Compute loss (MSE)
def compute_loss(predictions: Tensor, targets: Tensor) -> f24:
  squared_diffs = tree_squared_diff_net(predictions, targets)
  total = tree_sum_net(squared_diffs)
  count = tree_count_net(predictions)
  if count > 0:
    return total / count
  else:
    return 0.0

def tree_squared_diff_net(pred: Tensor, target: Tensor) -> Tensor:
  match pred:
    case Tensor/Node:
      match target:
        case Tensor/Node:
          return Tensor/Node {
            left: tree_squared_diff_net(pred.left, target.left),
            right: tree_squared_diff_net(pred.right, target.right)
          }
        case _:
          return Tensor/Nil
    case Tensor/Leaf:
      match target:
        case Tensor/Leaf:
          diff = pred.val - target.val
          return Tensor/Leaf { val: diff * diff }
        case _:
          return Tensor/Nil
    case Tensor/Nil:
      return Tensor/Nil

def tree_sum_net(t: Tensor) -> f24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return t.val
    case Tensor/Nil:
      return 0.0

def tree_count_net(t: Tensor) -> u24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return 1
    case Tensor/Nil:
      return 0

# ============================================================
# Model Summary
# ============================================================

# Count total parameters in network
def count_parameters(net: Network) -> u24:
  match net:
    case Network/Sequential:
      return count_layer_params(net.layers)
    case Network/Empty:
      return 0

def count_layer_params(layers: LayerList) -> u24:
  match layers:
    case LayerList/Cons:
      layer_params = count_single_layer_params(layers.head)
      return layer_params + count_layer_params(layers.tail)
    case LayerList/Nil:
      return 0

def count_single_layer_params(layer: Layer) -> u24:
  match layer:
    case Layer/LayerWithAct:
      match layer.dense:
        case DenseLayer/Dense:
          weights = layer.dense.input_size * layer.dense.output_size
          biases = layer.dense.output_size
          return weights + biases
