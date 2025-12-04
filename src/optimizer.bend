# Optimizers for neural network training
# Includes SGD, SGD with Momentum, Adam, RMSprop, and AdaGrad

type Tensor:
  Node { ~left: Tensor, ~right: Tensor }
  Leaf { val: f24 }
  Nil

# ============================================================
# Optimizer State Types
# ============================================================

# SGD with Momentum state
type MomentumState:
  Momentum {
    velocity: Tensor,      # v_t for weights
    bias_velocity: Tensor  # v_t for biases
  }
  Empty

# Adam optimizer state
type AdamState:
  Adam {
    m: Tensor,       # First moment (mean of gradients)
    v: Tensor,       # Second moment (variance of gradients)
    m_bias: Tensor,  # First moment for biases
    v_bias: Tensor,  # Second moment for biases
    t: u24           # Timestep
  }
  Empty

# RMSprop state
type RMSpropState:
  RMSprop {
    cache: Tensor,      # Running average of squared gradients
    cache_bias: Tensor  # For biases
  }
  Empty

# AdaGrad state
type AdaGradState:
  AdaGrad {
    accum: Tensor,      # Accumulated squared gradients
    accum_bias: Tensor  # For biases
  }
  Empty

# ============================================================
# SGD (Vanilla Stochastic Gradient Descent)
# ============================================================

# w = w - lr * gradient
def sgd_update(weights: Tensor, gradients: Tensor, lr: f24) -> Tensor:
  match weights:
    case Tensor/Node:
      match gradients:
        case Tensor/Node:
          return Tensor/Node {
            left: sgd_update(weights.left, gradients.left, lr),
            right: sgd_update(weights.right, gradients.right, lr)
          }
        case _:
          return weights
    case Tensor/Leaf:
      match gradients:
        case Tensor/Leaf:
          return Tensor/Leaf { val: weights.val - lr * gradients.val }
        case _:
          return weights
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# SGD with Momentum
# ============================================================

# v = momentum * v - lr * gradient
# w = w + v
def momentum_update(weights: Tensor, gradients: Tensor, velocity: Tensor, lr: f24, momentum: f24) -> (Tensor, Tensor):
  # Returns (new_weights, new_velocity)
  new_vel = update_velocity(velocity, gradients, lr, momentum)
  new_weights = apply_velocity(weights, new_vel)
  return (new_weights, new_vel)

def update_velocity(vel: Tensor, grad: Tensor, lr: f24, momentum: f24) -> Tensor:
  match vel:
    case Tensor/Node:
      match grad:
        case Tensor/Node:
          return Tensor/Node {
            left: update_velocity(vel.left, grad.left, lr, momentum),
            right: update_velocity(vel.right, grad.right, lr, momentum)
          }
        case _:
          return vel
    case Tensor/Leaf:
      match grad:
        case Tensor/Leaf:
          # v = momentum * v - lr * gradient
          return Tensor/Leaf { val: momentum * vel.val - lr * grad.val }
        case _:
          return vel
    case Tensor/Nil:
      return Tensor/Nil

def apply_velocity(weights: Tensor, velocity: Tensor) -> Tensor:
  match weights:
    case Tensor/Node:
      match velocity:
        case Tensor/Node:
          return Tensor/Node {
            left: apply_velocity(weights.left, velocity.left),
            right: apply_velocity(weights.right, velocity.right)
          }
        case _:
          return weights
    case Tensor/Leaf:
      match velocity:
        case Tensor/Leaf:
          return Tensor/Leaf { val: weights.val + velocity.val }
        case _:
          return weights
    case Tensor/Nil:
      return Tensor/Nil

# Initialize momentum state with zeros
def init_momentum_state(weights: Tensor, biases: Tensor) -> MomentumState:
  return MomentumState/Momentum {
    velocity: tree_zeros_like(weights),
    bias_velocity: tree_zeros_like(biases)
  }

# ============================================================
# Nesterov Accelerated Gradient (NAG)
# ============================================================

# Look-ahead gradient computation for Nesterov
# v = momentum * v - lr * gradient(w + momentum * v)
# w = w + v
def nesterov_update(weights: Tensor, gradients: Tensor, velocity: Tensor, lr: f24, momentum: f24) -> (Tensor, Tensor):
  # Update velocity with Nesterov correction
  new_vel = update_nesterov_velocity(velocity, gradients, lr, momentum)
  # Apply to weights with momentum correction
  new_weights = apply_nesterov(weights, velocity, new_vel, momentum)
  return (new_weights, new_vel)

def update_nesterov_velocity(vel: Tensor, grad: Tensor, lr: f24, momentum: f24) -> Tensor:
  match vel:
    case Tensor/Node:
      match grad:
        case Tensor/Node:
          return Tensor/Node {
            left: update_nesterov_velocity(vel.left, grad.left, lr, momentum),
            right: update_nesterov_velocity(vel.right, grad.right, lr, momentum)
          }
        case _:
          return vel
    case Tensor/Leaf:
      match grad:
        case Tensor/Leaf:
          return Tensor/Leaf { val: momentum * vel.val - lr * grad.val }
        case _:
          return vel
    case Tensor/Nil:
      return Tensor/Nil

def apply_nesterov(weights: Tensor, old_vel: Tensor, new_vel: Tensor, momentum: f24) -> Tensor:
  match weights:
    case Tensor/Node:
      match old_vel:
        case Tensor/Node:
          match new_vel:
            case Tensor/Node:
              return Tensor/Node {
                left: apply_nesterov(weights.left, old_vel.left, new_vel.left, momentum),
                right: apply_nesterov(weights.right, old_vel.right, new_vel.right, momentum)
              }
            case _:
              return weights
        case _:
          return weights
    case Tensor/Leaf:
      match old_vel:
        case Tensor/Leaf:
          match new_vel:
            case Tensor/Leaf:
              # w = w - momentum * v_old + (1 + momentum) * v_new
              return Tensor/Leaf { val: weights.val - momentum * old_vel.val + (1.0 + momentum) * new_vel.val }
            case _:
              return weights
        case _:
          return weights
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# Adam Optimizer
# ============================================================

# Adam: Adaptive Moment Estimation
# m = beta1 * m + (1 - beta1) * gradient
# v = beta2 * v + (1 - beta2) * gradient^2
# m_hat = m / (1 - beta1^t)
# v_hat = v / (1 - beta2^t)
# w = w - lr * m_hat / (sqrt(v_hat) + epsilon)

def adam_update(
  weights: Tensor,
  gradients: Tensor,
  m: Tensor,
  v: Tensor,
  t: u24,
  lr: f24,
  beta1: f24,
  beta2: f24,
  epsilon: f24
) -> (Tensor, Tensor, Tensor):
  # Returns (new_weights, new_m, new_v)
  new_m = update_first_moment(m, gradients, beta1)
  new_v = update_second_moment(v, gradients, beta2)

  # Bias correction
  beta1_t = pow_int(beta1, t)
  beta2_t = pow_int(beta2, t)
  m_correction = 1.0 / (1.0 - beta1_t)
  v_correction = 1.0 / (1.0 - beta2_t)

  new_weights = apply_adam_update(weights, new_m, new_v, lr, m_correction, v_correction, epsilon)
  return (new_weights, new_m, new_v)

# Update first moment estimate
def update_first_moment(m: Tensor, grad: Tensor, beta1: f24) -> Tensor:
  match m:
    case Tensor/Node:
      match grad:
        case Tensor/Node:
          return Tensor/Node {
            left: update_first_moment(m.left, grad.left, beta1),
            right: update_first_moment(m.right, grad.right, beta1)
          }
        case _:
          return m
    case Tensor/Leaf:
      match grad:
        case Tensor/Leaf:
          # m = beta1 * m + (1 - beta1) * g
          return Tensor/Leaf { val: beta1 * m.val + (1.0 - beta1) * grad.val }
        case _:
          return m
    case Tensor/Nil:
      return Tensor/Nil

# Update second moment estimate
def update_second_moment(v: Tensor, grad: Tensor, beta2: f24) -> Tensor:
  match v:
    case Tensor/Node:
      match grad:
        case Tensor/Node:
          return Tensor/Node {
            left: update_second_moment(v.left, grad.left, beta2),
            right: update_second_moment(v.right, grad.right, beta2)
          }
        case _:
          return v
    case Tensor/Leaf:
      match grad:
        case Tensor/Leaf:
          # v = beta2 * v + (1 - beta2) * g^2
          return Tensor/Leaf { val: beta2 * v.val + (1.0 - beta2) * grad.val * grad.val }
        case _:
          return v
    case Tensor/Nil:
      return Tensor/Nil

# Apply Adam weight update
def apply_adam_update(weights: Tensor, m: Tensor, v: Tensor, lr: f24, m_corr: f24, v_corr: f24, eps: f24) -> Tensor:
  match weights:
    case Tensor/Node:
      match m:
        case Tensor/Node:
          match v:
            case Tensor/Node:
              return Tensor/Node {
                left: apply_adam_update(weights.left, m.left, v.left, lr, m_corr, v_corr, eps),
                right: apply_adam_update(weights.right, m.right, v.right, lr, m_corr, v_corr, eps)
              }
            case _:
              return weights
        case _:
          return weights
    case Tensor/Leaf:
      match m:
        case Tensor/Leaf:
          match v:
            case Tensor/Leaf:
              m_hat = m.val * m_corr
              v_hat = v.val * v_corr
              # w = w - lr * m_hat / (sqrt(v_hat) + eps)
              return Tensor/Leaf { val: weights.val - lr * m_hat / (sqrt_approx(v_hat) + eps) }
            case _:
              return weights
        case _:
          return weights
    case Tensor/Nil:
      return Tensor/Nil

# Initialize Adam state
def init_adam_state(weights: Tensor, biases: Tensor) -> AdamState:
  return AdamState/Adam {
    m: tree_zeros_like(weights),
    v: tree_zeros_like(weights),
    m_bias: tree_zeros_like(biases),
    v_bias: tree_zeros_like(biases),
    t: 0
  }

# ============================================================
# AdamW (Adam with Weight Decay)
# ============================================================

# AdamW applies weight decay directly to weights, not through gradients
def adamw_update(
  weights: Tensor,
  gradients: Tensor,
  m: Tensor,
  v: Tensor,
  t: u24,
  lr: f24,
  beta1: f24,
  beta2: f24,
  epsilon: f24,
  weight_decay: f24
) -> (Tensor, Tensor, Tensor):
  new_m = update_first_moment(m, gradients, beta1)
  new_v = update_second_moment(v, gradients, beta2)

  beta1_t = pow_int(beta1, t)
  beta2_t = pow_int(beta2, t)
  m_correction = 1.0 / (1.0 - beta1_t)
  v_correction = 1.0 / (1.0 - beta2_t)

  new_weights = apply_adamw_update(weights, new_m, new_v, lr, m_correction, v_correction, epsilon, weight_decay)
  return (new_weights, new_m, new_v)

def apply_adamw_update(weights: Tensor, m: Tensor, v: Tensor, lr: f24, m_corr: f24, v_corr: f24, eps: f24, wd: f24) -> Tensor:
  match weights:
    case Tensor/Node:
      match m:
        case Tensor/Node:
          match v:
            case Tensor/Node:
              return Tensor/Node {
                left: apply_adamw_update(weights.left, m.left, v.left, lr, m_corr, v_corr, eps, wd),
                right: apply_adamw_update(weights.right, m.right, v.right, lr, m_corr, v_corr, eps, wd)
              }
            case _:
              return weights
        case _:
          return weights
    case Tensor/Leaf:
      match m:
        case Tensor/Leaf:
          match v:
            case Tensor/Leaf:
              m_hat = m.val * m_corr
              v_hat = v.val * v_corr
              # w = w - lr * (m_hat / (sqrt(v_hat) + eps) + wd * w)
              adam_update = m_hat / (sqrt_approx(v_hat) + eps)
              return Tensor/Leaf { val: weights.val - lr * (adam_update + wd * weights.val) }
            case _:
              return weights
        case _:
          return weights
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# RMSprop Optimizer
# ============================================================

# cache = decay * cache + (1 - decay) * gradient^2
# w = w - lr * gradient / (sqrt(cache) + epsilon)

def rmsprop_update(weights: Tensor, gradients: Tensor, cache: Tensor, lr: f24, decay: f24, epsilon: f24) -> (Tensor, Tensor):
  new_cache = update_rmsprop_cache(cache, gradients, decay)
  new_weights = apply_rmsprop(weights, gradients, new_cache, lr, epsilon)
  return (new_weights, new_cache)

def update_rmsprop_cache(cache: Tensor, grad: Tensor, decay: f24) -> Tensor:
  match cache:
    case Tensor/Node:
      match grad:
        case Tensor/Node:
          return Tensor/Node {
            left: update_rmsprop_cache(cache.left, grad.left, decay),
            right: update_rmsprop_cache(cache.right, grad.right, decay)
          }
        case _:
          return cache
    case Tensor/Leaf:
      match grad:
        case Tensor/Leaf:
          # cache = decay * cache + (1 - decay) * g^2
          return Tensor/Leaf { val: decay * cache.val + (1.0 - decay) * grad.val * grad.val }
        case _:
          return cache
    case Tensor/Nil:
      return Tensor/Nil

def apply_rmsprop(weights: Tensor, grad: Tensor, cache: Tensor, lr: f24, eps: f24) -> Tensor:
  match weights:
    case Tensor/Node:
      match grad:
        case Tensor/Node:
          match cache:
            case Tensor/Node:
              return Tensor/Node {
                left: apply_rmsprop(weights.left, grad.left, cache.left, lr, eps),
                right: apply_rmsprop(weights.right, grad.right, cache.right, lr, eps)
              }
            case _:
              return weights
        case _:
          return weights
    case Tensor/Leaf:
      match grad:
        case Tensor/Leaf:
          match cache:
            case Tensor/Leaf:
              # w = w - lr * g / (sqrt(cache) + eps)
              return Tensor/Leaf { val: weights.val - lr * grad.val / (sqrt_approx(cache.val) + eps) }
            case _:
              return weights
        case _:
          return weights
    case Tensor/Nil:
      return Tensor/Nil

# Initialize RMSprop state
def init_rmsprop_state(weights: Tensor, biases: Tensor) -> RMSpropState:
  return RMSpropState/RMSprop {
    cache: tree_zeros_like(weights),
    cache_bias: tree_zeros_like(biases)
  }

# ============================================================
# AdaGrad Optimizer
# ============================================================

# accum = accum + gradient^2
# w = w - lr * gradient / (sqrt(accum) + epsilon)

def adagrad_update(weights: Tensor, gradients: Tensor, accum: Tensor, lr: f24, epsilon: f24) -> (Tensor, Tensor):
  new_accum = update_adagrad_accum(accum, gradients)
  new_weights = apply_adagrad(weights, gradients, new_accum, lr, epsilon)
  return (new_weights, new_accum)

def update_adagrad_accum(accum: Tensor, grad: Tensor) -> Tensor:
  match accum:
    case Tensor/Node:
      match grad:
        case Tensor/Node:
          return Tensor/Node {
            left: update_adagrad_accum(accum.left, grad.left),
            right: update_adagrad_accum(accum.right, grad.right)
          }
        case _:
          return accum
    case Tensor/Leaf:
      match grad:
        case Tensor/Leaf:
          return Tensor/Leaf { val: accum.val + grad.val * grad.val }
        case _:
          return accum
    case Tensor/Nil:
      return Tensor/Nil

def apply_adagrad(weights: Tensor, grad: Tensor, accum: Tensor, lr: f24, eps: f24) -> Tensor:
  match weights:
    case Tensor/Node:
      match grad:
        case Tensor/Node:
          match accum:
            case Tensor/Node:
              return Tensor/Node {
                left: apply_adagrad(weights.left, grad.left, accum.left, lr, eps),
                right: apply_adagrad(weights.right, grad.right, accum.right, lr, eps)
              }
            case _:
              return weights
        case _:
          return weights
    case Tensor/Leaf:
      match grad:
        case Tensor/Leaf:
          match accum:
            case Tensor/Leaf:
              return Tensor/Leaf { val: weights.val - lr * grad.val / (sqrt_approx(accum.val) + eps) }
            case _:
              return weights
        case _:
          return weights
    case Tensor/Nil:
      return Tensor/Nil

# Initialize AdaGrad state
def init_adagrad_state(weights: Tensor, biases: Tensor) -> AdaGradState:
  return AdaGradState/AdaGrad {
    accum: tree_zeros_like(weights),
    accum_bias: tree_zeros_like(biases)
  }

# ============================================================
# Gradient Clipping
# ============================================================

# Clip gradients by value
def clip_by_value(grad: Tensor, min_val: f24, max_val: f24) -> Tensor:
  match grad:
    case Tensor/Node:
      return Tensor/Node {
        left: clip_by_value(grad.left, min_val, max_val),
        right: clip_by_value(grad.right, min_val, max_val)
      }
    case Tensor/Leaf:
      g = grad.val
      if g < min_val:
        return Tensor/Leaf { val: min_val }
      else:
        if g > max_val:
          return Tensor/Leaf { val: max_val }
        else:
          return Tensor/Leaf { val: g }
    case Tensor/Nil:
      return Tensor/Nil

# Compute L2 norm of tensor
def tensor_norm(t: Tensor) -> f24:
  sum_sq = tensor_sum_squares(t)
  return sqrt_approx(sum_sq)

def tensor_sum_squares(t: Tensor) -> f24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return t.val * t.val
    case Tensor/Nil:
      return 0.0

# Clip gradients by global norm
def clip_by_norm(grad: Tensor, max_norm: f24) -> Tensor:
  norm = tensor_norm(grad)
  if norm > max_norm:
    scale = max_norm / norm
    return tree_scale_opt(grad, scale)
  else:
    return grad

def tree_scale_opt(t: Tensor, s: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_scale_opt(t.left, s),
        right: tree_scale_opt(t.right, s)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: t.val * s }
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# Helper Functions
# ============================================================

# Create tensor of zeros with same shape
def tree_zeros_like(t: Tensor) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_zeros_like(t.left),
        right: tree_zeros_like(t.right)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: 0.0 }
    case Tensor/Nil:
      return Tensor/Nil

# Square root approximation using Newton-Raphson
def sqrt_approx(x: f24) -> f24:
  if x < 0.0001:
    return 0.0
  else:
    # Initial guess
    guess = x * 0.5
    # Newton-Raphson iterations
    guess = 0.5 * (guess + x / guess)
    guess = 0.5 * (guess + x / guess)
    guess = 0.5 * (guess + x / guess)
    return guess

# Integer power for bias correction
def pow_int(base: f24, exp: u24) -> f24:
  if exp == 0:
    return 1.0
  else:
    result = 1.0
    bend i=0, r=1.0, b=base:
      when i < exp:
        fork(i + 1, r * b, b)
      else:
        result = r
    return result
