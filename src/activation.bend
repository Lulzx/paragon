# Activation functions for neural networks
# All operations are designed for parallel execution

type Tensor:
  Node { ~left: Tensor, ~right: Tensor }
  Leaf { val: f24 }
  Nil

# ============================================================
# Activation Functions
# ============================================================

# ReLU activation: max(0, x)
def relu(x: f24) -> f24:
  if x > 0.0:
    return x
  else:
    return 0.0

# ReLU derivative: 1 if x > 0, else 0
def relu_deriv(x: f24) -> f24:
  if x > 0.0:
    return 1.0
  else:
    return 0.0

# Sigmoid approximation: 1 / (1 + e^(-x))
# Using piecewise linear approximation for performance
def sigmoid(x: f24) -> f24:
  if x > 5.0:
    return 1.0
  else:
    if x < -5.0:
      return 0.0
    else:
      # Linear approximation in the middle range
      return 0.5 + x * 0.1

# Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
def sigmoid_deriv(x: f24) -> f24:
  s = sigmoid(x)
  return s * (1.0 - s)

# Tanh approximation
def tanh_approx(x: f24) -> f24:
  if x > 3.0:
    return 1.0
  else:
    if x < -3.0:
      return -1.0
    else:
      # Approximation: x * (27 + x^2) / (27 + 9*x^2)
      x2 = x * x
      return x * (27.0 + x2) / (27.0 + 9.0 * x2)

# Leaky ReLU: max(0.01*x, x)
def leaky_relu(x: f24) -> f24:
  if x > 0.0:
    return x
  else:
    return 0.01 * x

# ============================================================
# Apply Activation to Tree (Parallel)
# ============================================================

# Apply ReLU to entire tree
def tree_relu(t: Tensor) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_relu(t.left),
        right: tree_relu(t.right)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: relu(t.val) }
    case Tensor/Nil:
      return Tensor/Nil

# Apply ReLU derivative to tree
def tree_relu_deriv(t: Tensor) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_relu_deriv(t.left),
        right: tree_relu_deriv(t.right)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: relu_deriv(t.val) }
    case Tensor/Nil:
      return Tensor/Nil

# Apply sigmoid to entire tree
def tree_sigmoid(t: Tensor) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_sigmoid(t.left),
        right: tree_sigmoid(t.right)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: sigmoid(t.val) }
    case Tensor/Nil:
      return Tensor/Nil

# Apply sigmoid derivative to tree
def tree_sigmoid_deriv(t: Tensor) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_sigmoid_deriv(t.left),
        right: tree_sigmoid_deriv(t.right)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: sigmoid_deriv(t.val) }
    case Tensor/Nil:
      return Tensor/Nil

# Apply tanh to entire tree
def tree_tanh(t: Tensor) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_tanh(t.left),
        right: tree_tanh(t.right)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: tanh_approx(t.val) }
    case Tensor/Nil:
      return Tensor/Nil

# Apply leaky ReLU to entire tree
def tree_leaky_relu(t: Tensor) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_leaky_relu(t.left),
        right: tree_leaky_relu(t.right)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: leaky_relu(t.val) }
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# Softmax (for classification output)
# ============================================================

# Compute max of tree (for numerical stability)
def tree_max(t: Tensor) -> f24:
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

# Compute exp of each element minus max (for stability)
def tree_exp_shifted(t: Tensor, max_val: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_exp_shifted(t.left, max_val),
        right: tree_exp_shifted(t.right, max_val)
      }
    case Tensor/Leaf:
      # Approximation of exp using Taylor series
      x = t.val - max_val
      return Tensor/Leaf { val: 1.0 + x + x*x*0.5 + x*x*x*0.166667 }
    case Tensor/Nil:
      return Tensor/Nil
