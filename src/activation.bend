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
      return Tensor/Leaf { val: exp_approx(x) }
    case Tensor/Nil:
      return Tensor/Nil

# Exponential approximation using Taylor series
# exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120
def exp_approx(x: f24) -> f24:
  # Clamp to avoid overflow
  if x > 10.0:
    return 22026.0  # exp(10)
  else:
    if x < -10.0:
      return 0.00005  # exp(-10)
    else:
      # Taylor series expansion
      x2 = x * x
      x3 = x2 * x
      x4 = x3 * x
      x5 = x4 * x
      return 1.0 + x + x2 * 0.5 + x3 * 0.166667 + x4 * 0.041667 + x5 * 0.008333

# Sum all elements in tree (for softmax normalization)
def tree_sum_act(t: Tensor) -> f24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return t.val
    case Tensor/Nil:
      return 0.0

# Divide each element by sum (for softmax normalization)
def tree_divide(t: Tensor, divisor: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_divide(t.left, divisor),
        right: tree_divide(t.right, divisor)
      }
    case Tensor/Leaf:
      if divisor > 0.0001:
        return Tensor/Leaf { val: t.val / divisor }
      else:
        return Tensor/Leaf { val: 0.0 }
    case Tensor/Nil:
      return Tensor/Nil

# Softmax activation (numerically stable)
# softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
def tree_softmax(t: Tensor) -> Tensor:
  max_val = tree_max(t)
  exp_shifted = tree_exp_shifted(t, max_val)
  sum_exp = tree_sum_act(exp_shifted)
  return tree_divide(exp_shifted, sum_exp)

# Softmax derivative for backpropagation
# For softmax output s_i, gradient is: s_i * (1 - s_i) for diagonal
# Full Jacobian: diag(s) - s * s^T
def tree_softmax_deriv(softmax_output: Tensor, output_grad: Tensor) -> Tensor:
  # Simplified: element-wise s * (1 - s) * grad
  # Full Jacobian would require matrix operations
  match softmax_output:
    case Tensor/Node:
      match output_grad:
        case Tensor/Node:
          return Tensor/Node {
            left: tree_softmax_deriv(softmax_output.left, output_grad.left),
            right: tree_softmax_deriv(softmax_output.right, output_grad.right)
          }
        case _:
          return Tensor/Nil
    case Tensor/Leaf:
      match output_grad:
        case Tensor/Leaf:
          s = softmax_output.val
          return Tensor/Leaf { val: output_grad.val * s * (1.0 - s) }
        case _:
          return Tensor/Nil
    case Tensor/Nil:
      return Tensor/Nil
