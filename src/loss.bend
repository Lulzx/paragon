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

# ============================================================
# Categorical Cross-Entropy Loss (for multi-class classification)
# ============================================================

# Cross-entropy element: -target * log(prediction)
# Used with softmax output where predictions sum to 1
def cross_entropy_element(pred: f24, target: f24) -> f24:
  # Only contributes if target is 1 (one-hot encoding)
  return -target * safe_log(pred)

# Apply cross-entropy to tree (for one-hot encoded targets)
def tree_cross_entropy(pred: Tensor, target: Tensor) -> Tensor:
  match pred:
    case Tensor/Node:
      match target:
        case Tensor/Node:
          return Tensor/Node {
            left: tree_cross_entropy(pred.left, target.left),
            right: tree_cross_entropy(pred.right, target.right)
          }
        case _:
          return Tensor/Nil
    case Tensor/Leaf:
      match target:
        case Tensor/Leaf:
          return Tensor/Leaf { val: cross_entropy_element(pred.val, target.val) }
        case _:
          return Tensor/Nil
    case Tensor/Nil:
      return Tensor/Nil

# Categorical cross-entropy loss
# L = -sum(target * log(pred))
def cross_entropy_loss(predictions: Tensor, targets: Tensor) -> f24:
  ce_values = tree_cross_entropy(predictions, targets)
  return tree_sum_local(ce_values)

# Cross-entropy gradient with softmax: (prediction - target)
# This is the combined gradient of softmax + cross-entropy
# Much simpler and more numerically stable than separate gradients
def cross_entropy_gradient(predictions: Tensor, targets: Tensor) -> Tensor:
  return tree_diff_simple(predictions, targets)

# Simple difference: pred - target
def tree_diff_simple(pred: Tensor, target: Tensor) -> Tensor:
  match pred:
    case Tensor/Node:
      match target:
        case Tensor/Node:
          return Tensor/Node {
            left: tree_diff_simple(pred.left, target.left),
            right: tree_diff_simple(pred.right, target.right)
          }
        case _:
          return Tensor/Nil
    case Tensor/Leaf:
      match target:
        case Tensor/Leaf:
          return Tensor/Leaf { val: pred.val - target.val }
        case _:
          return Tensor/Nil
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# Sparse Cross-Entropy (for integer class labels)
# ============================================================

# Get the predicted probability for a specific class index
# This is useful when targets are integer class labels rather than one-hot
def get_class_prob(predictions: Tensor, class_idx: u24) -> f24:
  return tree_get_at_index(predictions, class_idx, 0)

# Helper to get value at index in tree
def tree_get_at_index(t: Tensor, target_idx: u24, current_idx: u24) -> f24:
  match t:
    case Tensor/Node:
      # In a balanced tree, left subtree has lower indices
      left_result = tree_get_at_index(t.left, target_idx, current_idx)
      if left_result > -999998.0:
        return left_result
      else:
        # Count elements in left subtree and continue to right
        left_count = tree_count(t.left)
        return tree_get_at_index(t.right, target_idx, current_idx + left_count)
    case Tensor/Leaf:
      if current_idx == target_idx:
        return t.val
      else:
        return -999999.0  # Sentinel for not found
    case Tensor/Nil:
      return -999999.0

# Sparse cross-entropy loss (target is class index, not one-hot)
def sparse_cross_entropy_loss(predictions: Tensor, class_idx: u24) -> f24:
  prob = get_class_prob(predictions, class_idx)
  return -safe_log(prob)

# ============================================================
# Label Smoothing Cross-Entropy
# ============================================================

# Apply label smoothing to one-hot targets
# smoothed = (1 - epsilon) * one_hot + epsilon / num_classes
def smooth_labels(targets: Tensor, epsilon: f24, num_classes: u24) -> Tensor:
  smooth_val = epsilon / num_classes
  scale = 1.0 - epsilon
  return tree_smooth_labels(targets, scale, smooth_val)

def tree_smooth_labels(t: Tensor, scale: f24, smooth_val: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_smooth_labels(t.left, scale, smooth_val),
        right: tree_smooth_labels(t.right, scale, smooth_val)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: t.val * scale + smooth_val }
    case Tensor/Nil:
      return Tensor/Nil

# Label smoothing cross-entropy loss
def label_smoothing_loss(predictions: Tensor, targets: Tensor, epsilon: f24, num_classes: u24) -> f24:
  smoothed = smooth_labels(targets, epsilon, num_classes)
  return cross_entropy_loss(predictions, smoothed)

# ============================================================
# Focal Loss (for imbalanced classification)
# ============================================================

# Focal loss element: -(1-p)^gamma * target * log(p)
# Down-weights easy examples, focuses on hard examples
def focal_element(pred: f24, target: f24, gamma: f24) -> f24:
  if target > 0.5:  # This is the positive class
    # -(1-p)^gamma * log(p)
    pt = pred
    weight = pow_approx(1.0 - pt, gamma)
    return -weight * safe_log(pt)
  else:
    # -(p)^gamma * log(1-p) for negative class
    pt = 1.0 - pred
    weight = pow_approx(pred, gamma)
    return -weight * safe_log(pt)

# Power approximation using exp and log
def pow_approx(base: f24, exp: f24) -> f24:
  if base < 0.0001:
    return 0.0
  else:
    # x^y = exp(y * log(x))
    log_base = safe_log(base)
    return exp_approx_loss(exp * log_base)

# Exponential approximation for loss calculations
def exp_approx_loss(x: f24) -> f24:
  if x > 10.0:
    return 22026.0
  else:
    if x < -10.0:
      return 0.00005
    else:
      x2 = x * x
      x3 = x2 * x
      x4 = x3 * x
      return 1.0 + x + x2 * 0.5 + x3 * 0.166667 + x4 * 0.041667

# Apply focal loss to tree
def tree_focal(pred: Tensor, target: Tensor, gamma: f24) -> Tensor:
  match pred:
    case Tensor/Node:
      match target:
        case Tensor/Node:
          return Tensor/Node {
            left: tree_focal(pred.left, target.left, gamma),
            right: tree_focal(pred.right, target.right, gamma)
          }
        case _:
          return Tensor/Nil
    case Tensor/Leaf:
      match target:
        case Tensor/Leaf:
          return Tensor/Leaf { val: focal_element(pred.val, target.val, gamma) }
        case _:
          return Tensor/Nil
    case Tensor/Nil:
      return Tensor/Nil

# Focal loss (for class imbalance)
def focal_loss(predictions: Tensor, targets: Tensor, gamma: f24) -> f24:
  focal_values = tree_focal(predictions, targets, gamma)
  return tree_sum_local(focal_values)
