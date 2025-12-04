# Loss functions for neural network training
# Supports Mean Squared Error and Cross-Entropy

type Tensor:
  Node { ~left: Tensor, ~right: Tensor }
  Leaf { val: f24 }
  Nil

# ============================================================
# Mean Squared Error (MSE) Loss
# ============================================================

# Compute (prediction - target)^2 element-wise
def tree_squared_diff(pred: Tensor, target: Tensor) -> Tensor:
  match pred:
    case Tensor/Node:
      match target:
        case Tensor/Node:
          return Tensor/Node {
            left: tree_squared_diff(pred.left, target.left),
            right: tree_squared_diff(pred.right, target.right)
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

# Sum all elements in tree (parallel reduction)
def tree_sum_local(t: Tensor) -> f24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return t.val
    case Tensor/Nil:
      return 0.0

# Count elements in tree
def tree_count(t: Tensor) -> u24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return 1
    case Tensor/Nil:
      return 0

# Mean Squared Error loss
def mse_loss(predictions: Tensor, targets: Tensor) -> f24:
  squared_diffs = tree_squared_diff(predictions, targets)
  total = tree_sum_local(squared_diffs)
  count = tree_count(predictions)
  if count > 0:
    return total / count
  else:
    return 0.0

# MSE gradient: 2 * (prediction - target) / n
def mse_gradient(predictions: Tensor, targets: Tensor) -> Tensor:
  count = tree_count(predictions)
  scale = 2.0 / count
  return tree_diff_scaled(predictions, targets, scale)

# Compute scaled difference
def tree_diff_scaled(pred: Tensor, target: Tensor, scale: f24) -> Tensor:
  match pred:
    case Tensor/Node:
      match target:
        case Tensor/Node:
          return Tensor/Node {
            left: tree_diff_scaled(pred.left, target.left, scale),
            right: tree_diff_scaled(pred.right, target.right, scale)
          }
        case _:
          return Tensor/Nil
    case Tensor/Leaf:
      match target:
        case Tensor/Leaf:
          return Tensor/Leaf { val: (pred.val - target.val) * scale }
        case _:
          return Tensor/Nil
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# Binary Cross-Entropy Loss
# ============================================================

# Safe log approximation to avoid log(0)
def safe_log(x: f24) -> f24:
  if x < 0.0001:
    return -9.21  # log(0.0001)
  else:
    if x > 0.9999:
      return 0.0  # log(1) ≈ 0
    else:
      # Taylor series approximation for log(x) around x=1
      # log(x) ≈ (x-1) - (x-1)^2/2 + (x-1)^3/3 - ...
      t = x - 1.0
      return t - t*t*0.5 + t*t*t*0.333

# Binary cross-entropy element-wise
def bce_element(pred: f24, target: f24) -> f24:
  # -[y*log(p) + (1-y)*log(1-p)]
  log_pred = safe_log(pred)
  log_one_minus_pred = safe_log(1.0 - pred)
  return -(target * log_pred + (1.0 - target) * log_one_minus_pred)

# Apply BCE to tree
def tree_bce(pred: Tensor, target: Tensor) -> Tensor:
  match pred:
    case Tensor/Node:
      match target:
        case Tensor/Node:
          return Tensor/Node {
            left: tree_bce(pred.left, target.left),
            right: tree_bce(pred.right, target.right)
          }
        case _:
          return Tensor/Nil
    case Tensor/Leaf:
      match target:
        case Tensor/Leaf:
          return Tensor/Leaf { val: bce_element(pred.val, target.val) }
        case _:
          return Tensor/Nil
    case Tensor/Nil:
      return Tensor/Nil

# Binary Cross-Entropy loss
def bce_loss(predictions: Tensor, targets: Tensor) -> f24:
  bce_values = tree_bce(predictions, targets)
  total = tree_sum_local(bce_values)
  count = tree_count(predictions)
  if count > 0:
    return total / count
  else:
    return 0.0

# BCE gradient: (prediction - target) / (prediction * (1 - prediction))
def bce_gradient(predictions: Tensor, targets: Tensor) -> Tensor:
  count = tree_count(predictions)
  return tree_bce_grad(predictions, targets, count)

def tree_bce_grad(pred: Tensor, target: Tensor, count: u24) -> Tensor:
  match pred:
    case Tensor/Node:
      match target:
        case Tensor/Node:
          return Tensor/Node {
            left: tree_bce_grad(pred.left, target.left, count),
            right: tree_bce_grad(pred.right, target.right, count)
          }
        case _:
          return Tensor/Nil
    case Tensor/Leaf:
      match target:
        case Tensor/Leaf:
          # Gradient: (p - y) / (p * (1-p) * n)
          p = pred.val
          y = target.val
          denom = p * (1.0 - p) * count
          if denom < 0.0001:
            return Tensor/Leaf { val: 0.0 }
          else:
            return Tensor/Leaf { val: (p - y) / denom }
        case _:
          return Tensor/Nil
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# Huber Loss (smooth L1)
# ============================================================

# Huber loss element: L2 for small errors, L1 for large
def huber_element(pred: f24, target: f24, delta: f24) -> f24:
  diff = pred - target
  abs_diff = if diff > 0.0 { diff } else { -diff }
  if abs_diff <= delta:
    return 0.5 * diff * diff
  else:
    return delta * (abs_diff - 0.5 * delta)

# Huber loss on tree
def tree_huber(pred: Tensor, target: Tensor, delta: f24) -> Tensor:
  match pred:
    case Tensor/Node:
      match target:
        case Tensor/Node:
          return Tensor/Node {
            left: tree_huber(pred.left, target.left, delta),
            right: tree_huber(pred.right, target.right, delta)
          }
        case _:
          return Tensor/Nil
    case Tensor/Leaf:
      match target:
        case Tensor/Leaf:
          return Tensor/Leaf { val: huber_element(pred.val, target.val, delta) }
        case _:
          return Tensor/Nil
    case Tensor/Nil:
      return Tensor/Nil

# Huber loss
def huber_loss(predictions: Tensor, targets: Tensor, delta: f24) -> f24:
  huber_values = tree_huber(predictions, targets, delta)
  total = tree_sum_local(huber_values)
  count = tree_count(predictions)
  if count > 0:
    return total / count
  else:
    return 0.0
