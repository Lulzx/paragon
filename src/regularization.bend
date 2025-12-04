# Regularization techniques for neural networks
# Includes Dropout, L1/L2 regularization, and Batch Normalization

type Tensor:
  Node { ~left: Tensor, ~right: Tensor }
  Leaf { val: f24 }
  Nil

# ============================================================
# Dropout
# ============================================================

# Dropout state for tracking random masks
type DropoutMask:
  Mask { data: Tensor }
  None

# Apply dropout during training
# Randomly sets elements to 0 with probability p
# Scales remaining elements by 1/(1-p) for inverted dropout
def apply_dropout(t: Tensor, mask: Tensor, keep_prob: f24) -> Tensor:
  scale = 1.0 / keep_prob
  match t:
    case Tensor/Node:
      match mask:
        case Tensor/Node:
          return Tensor/Node {
            left: apply_dropout(t.left, mask.left, keep_prob),
            right: apply_dropout(t.right, mask.right, keep_prob)
          }
        case _:
          return t
    case Tensor/Leaf:
      match mask:
        case Tensor/Leaf:
          # If mask > keep_prob, drop (set to 0)
          # Otherwise, scale by 1/(1-p)
          if mask.val > keep_prob:
            return Tensor/Leaf { val: 0.0 }
          else:
            return Tensor/Leaf { val: t.val * scale }
        case _:
          return t
    case Tensor/Nil:
      return Tensor/Nil

# Generate dropout mask using LCG random
def generate_dropout_mask(size: u24, seed: u24) -> Tensor:
  bend i=0, s=seed:
    when i < size:
      # Generate pseudo-random value in [0, 1]
      rand_val = lcg_random(s)
      result = Tensor/Node {
        left: Tensor/Leaf { val: rand_val },
        right: fork(i + 1, s + 1)
      }
    else:
      result = Tensor/Nil
  return result

# Linear congruential generator for pseudo-random numbers
def lcg_random(seed: u24) -> f24:
  a = 1103515245
  c = 12345
  m = 2147483648
  next_val = (a * seed + c) % m
  return next_val / m

# Dropout backward pass (apply same mask to gradients)
def dropout_backward(grad: Tensor, mask: Tensor, keep_prob: f24) -> Tensor:
  # Same operation as forward - mask out same positions
  return apply_dropout(grad, mask, keep_prob)

# ============================================================
# Spatial Dropout (for 2D feature maps)
# ============================================================

# Drops entire feature channels instead of individual elements
# Useful for convolutional networks (when we add conv layers)
def apply_spatial_dropout(t: Tensor, channel_mask: Tensor) -> Tensor:
  match t:
    case Tensor/Node:
      match channel_mask:
        case Tensor/Node:
          return Tensor/Node {
            left: apply_spatial_dropout(t.left, channel_mask.left),
            right: apply_spatial_dropout(t.right, channel_mask.right)
          }
        case Tensor/Leaf:
          # Apply same mask value to entire subtree
          if channel_mask.val > 0.5:
            return tree_zeros_copy(t)
          else:
            return t
        case _:
          return t
    case Tensor/Leaf:
      match channel_mask:
        case Tensor/Leaf:
          if channel_mask.val > 0.5:
            return Tensor/Leaf { val: 0.0 }
          else:
            return t
        case _:
          return t
    case Tensor/Nil:
      return Tensor/Nil

def tree_zeros_copy(t: Tensor) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_zeros_copy(t.left),
        right: tree_zeros_copy(t.right)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: 0.0 }
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# L1 Regularization (Lasso)
# ============================================================

# L1 penalty: lambda * sum(|w|)
def l1_penalty(weights: Tensor, lambda_l1: f24) -> f24:
  sum_abs = tree_sum_abs(weights)
  return lambda_l1 * sum_abs

def tree_sum_abs(t: Tensor) -> f24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      if t.val > 0.0:
        return t.val
      else:
        return -t.val
    case Tensor/Nil:
      return 0.0

# L1 gradient: lambda * sign(w)
def l1_gradient(weights: Tensor, lambda_l1: f24) -> Tensor:
  match weights:
    case Tensor/Node:
      return Tensor/Node {
        left: l1_gradient(weights.left, lambda_l1),
        right: l1_gradient(weights.right, lambda_l1)
      }
    case Tensor/Leaf:
      if weights.val > 0.0:
        return Tensor/Leaf { val: lambda_l1 }
      else:
        if weights.val < 0.0:
          return Tensor/Leaf { val: -lambda_l1 }
        else:
          return Tensor/Leaf { val: 0.0 }
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# L2 Regularization (Ridge / Weight Decay)
# ============================================================

# L2 penalty: 0.5 * lambda * sum(w^2)
def l2_penalty(weights: Tensor, lambda_l2: f24) -> f24:
  sum_sq = tree_sum_squares_reg(weights)
  return 0.5 * lambda_l2 * sum_sq

def tree_sum_squares_reg(t: Tensor) -> f24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return t.val * t.val
    case Tensor/Nil:
      return 0.0

# L2 gradient: lambda * w
def l2_gradient(weights: Tensor, lambda_l2: f24) -> Tensor:
  match weights:
    case Tensor/Node:
      return Tensor/Node {
        left: l2_gradient(weights.left, lambda_l2),
        right: l2_gradient(weights.right, lambda_l2)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: lambda_l2 * weights.val }
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# Elastic Net (L1 + L2 combined)
# ============================================================

# Elastic net penalty: alpha * L1 + (1-alpha) * L2
def elastic_net_penalty(weights: Tensor, lambda_reg: f24, alpha: f24) -> f24:
  l1 = l1_penalty(weights, lambda_reg * alpha)
  l2 = l2_penalty(weights, lambda_reg * (1.0 - alpha))
  return l1 + l2

# Elastic net gradient
def elastic_net_gradient(weights: Tensor, lambda_reg: f24, alpha: f24) -> Tensor:
  l1_grad = l1_gradient(weights, lambda_reg * alpha)
  l2_grad = l2_gradient(weights, lambda_reg * (1.0 - alpha))
  return tree_add_reg(l1_grad, l2_grad)

def tree_add_reg(a: Tensor, b: Tensor) -> Tensor:
  match a:
    case Tensor/Node:
      match b:
        case Tensor/Node:
          return Tensor/Node {
            left: tree_add_reg(a.left, b.left),
            right: tree_add_reg(a.right, b.right)
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
# Batch Normalization
# ============================================================

# BatchNorm state
type BatchNormState:
  BN {
    gamma: Tensor,          # Scale parameter (learnable)
    beta: Tensor,           # Shift parameter (learnable)
    running_mean: Tensor,   # Running mean for inference
    running_var: Tensor,    # Running variance for inference
    momentum: f24           # Momentum for running stats
  }
  Empty

# Initialize batch normalization parameters
def init_batchnorm(size: u24) -> BatchNormState:
  return BatchNormState/BN {
    gamma: tree_ones(size),
    beta: tree_zeros(size),
    running_mean: tree_zeros(size),
    running_var: tree_ones(size),
    momentum: 0.1
  }

def tree_ones(size: u24) -> Tensor:
  bend i=0:
    when i < size:
      result = Tensor/Node {
        left: Tensor/Leaf { val: 1.0 },
        right: fork(i + 1)
      }
    else:
      result = Tensor/Nil
  return result

def tree_zeros(size: u24) -> Tensor:
  bend i=0:
    when i < size:
      result = Tensor/Node {
        left: Tensor/Leaf { val: 0.0 },
        right: fork(i + 1)
      }
    else:
      result = Tensor/Nil
  return result

# Compute mean of tensor
def tensor_mean(t: Tensor) -> f24:
  total = tree_sum_bn(t)
  count = tree_count_bn(t)
  if count > 0:
    return total / count
  else:
    return 0.0

def tree_sum_bn(t: Tensor) -> f24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return t.val
    case Tensor/Nil:
      return 0.0

def tree_count_bn(t: Tensor) -> u24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return 1
    case Tensor/Nil:
      return 0

# Compute variance of tensor
def tensor_variance(t: Tensor, mean: f24) -> f24:
  sum_sq_diff = tree_sum_sq_diff(t, mean)
  count = tree_count_bn(t)
  if count > 0:
    return sum_sq_diff / count
  else:
    return 1.0

def tree_sum_sq_diff(t: Tensor, mean: f24) -> f24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      diff = t.val - mean
      return diff * diff
    case Tensor/Nil:
      return 0.0

# Normalize tensor: (x - mean) / sqrt(var + epsilon)
def normalize_tensor(t: Tensor, mean: f24, var: f24, epsilon: f24) -> Tensor:
  std = sqrt_approx_bn(var + epsilon)
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: normalize_tensor(t.left, mean, var, epsilon),
        right: normalize_tensor(t.right, mean, var, epsilon)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: (t.val - mean) / std }
    case Tensor/Nil:
      return Tensor/Nil

# Square root approximation
def sqrt_approx_bn(x: f24) -> f24:
  if x < 0.0001:
    return 0.01
  else:
    guess = x * 0.5
    guess = 0.5 * (guess + x / guess)
    guess = 0.5 * (guess + x / guess)
    guess = 0.5 * (guess + x / guess)
    return guess

# Apply scale and shift: gamma * x_norm + beta
def scale_and_shift(x_norm: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
  match x_norm:
    case Tensor/Node:
      match gamma:
        case Tensor/Node:
          match beta:
            case Tensor/Node:
              return Tensor/Node {
                left: scale_and_shift(x_norm.left, gamma.left, beta.left),
                right: scale_and_shift(x_norm.right, gamma.right, beta.right)
              }
            case _:
              return x_norm
        case _:
          return x_norm
    case Tensor/Leaf:
      match gamma:
        case Tensor/Leaf:
          match beta:
            case Tensor/Leaf:
              return Tensor/Leaf { val: gamma.val * x_norm.val + beta.val }
            case _:
              return x_norm
        case _:
          return x_norm
    case Tensor/Nil:
      return Tensor/Nil

# Batch normalization forward pass (training mode)
def batchnorm_forward_train(x: Tensor, state: BatchNormState, epsilon: f24) -> (Tensor, f24, f24):
  # Returns (normalized output, batch_mean, batch_var)
  match state:
    case BatchNormState/BN:
      batch_mean = tensor_mean(x)
      batch_var = tensor_variance(x, batch_mean)
      x_norm = normalize_tensor(x, batch_mean, batch_var, epsilon)
      output = scale_and_shift(x_norm, state.gamma, state.beta)
      return (output, batch_mean, batch_var)
    case BatchNormState/Empty:
      return (x, 0.0, 1.0)

# Batch normalization forward pass (inference mode)
def batchnorm_forward_inference(x: Tensor, state: BatchNormState, epsilon: f24) -> Tensor:
  match state:
    case BatchNormState/BN:
      x_norm = normalize_tensor(x, tensor_mean(state.running_mean), tensor_mean(state.running_var), epsilon)
      return scale_and_shift(x_norm, state.gamma, state.beta)
    case BatchNormState/Empty:
      return x

# Update running statistics
def update_running_stats(state: BatchNormState, batch_mean: f24, batch_var: f24) -> BatchNormState:
  match state:
    case BatchNormState/BN:
      new_running_mean = update_running_stat(state.running_mean, batch_mean, state.momentum)
      new_running_var = update_running_stat(state.running_var, batch_var, state.momentum)
      return BatchNormState/BN {
        gamma: state.gamma,
        beta: state.beta,
        running_mean: new_running_mean,
        running_var: new_running_var,
        momentum: state.momentum
      }
    case BatchNormState/Empty:
      return BatchNormState/Empty

def update_running_stat(running: Tensor, batch_val: f24, momentum: f24) -> Tensor:
  # running = (1 - momentum) * running + momentum * batch_val
  match running:
    case Tensor/Node:
      return Tensor/Node {
        left: update_running_stat(running.left, batch_val, momentum),
        right: update_running_stat(running.right, batch_val, momentum)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: (1.0 - momentum) * running.val + momentum * batch_val }
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# Layer Normalization (alternative to BatchNorm)
# ============================================================

# Layer norm normalizes across features for each sample
# (useful when batch size is small or 1)
def layernorm_forward(x: Tensor, gamma: Tensor, beta: Tensor, epsilon: f24) -> Tensor:
  mean = tensor_mean(x)
  var = tensor_variance(x, mean)
  x_norm = normalize_tensor(x, mean, var, epsilon)
  return scale_and_shift(x_norm, gamma, beta)

# ============================================================
# Weight Normalization
# ============================================================

# Reparameterize weights as w = g * v / ||v||
def weight_normalize(v: Tensor, g: f24) -> Tensor:
  norm = tensor_norm_wn(v)
  if norm < 0.0001:
    return v
  else:
    return tree_scale_wn(v, g / norm)

def tensor_norm_wn(t: Tensor) -> f24:
  sum_sq = tree_sum_squares_reg(t)
  return sqrt_approx_bn(sum_sq)

def tree_scale_wn(t: Tensor, s: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_scale_wn(t.left, s),
        right: tree_scale_wn(t.right, s)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: t.val * s }
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# Max Norm Constraint
# ============================================================

# Constrain weights so ||w|| <= max_norm
def max_norm_constraint(weights: Tensor, max_norm: f24) -> Tensor:
  norm = tensor_norm_wn(weights)
  if norm > max_norm:
    scale = max_norm / norm
    return tree_scale_wn(weights, scale)
  else:
    return weights
