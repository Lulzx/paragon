# MNIST Training Example
# Demonstrates training a neural network on MNIST digit classification

type Tensor:
  Node { ~left: Tensor, ~right: Tensor }
  Leaf { val: f24 }
  Nil

# ============================================================
# Network Architecture
# ============================================================

# MNIST classifier: 784 -> 128 -> 64 -> 10
# - Input: 784 (28x28 flattened)
# - Hidden 1: 128 neurons with ReLU
# - Hidden 2: 64 neurons with ReLU
# - Output: 10 neurons with Softmax

type MNISTNetwork:
  Net {
    w1: Tensor,  # 784 x 128 weights
    b1: Tensor,  # 128 biases
    w2: Tensor,  # 128 x 64 weights
    b2: Tensor,  # 64 biases
    w3: Tensor,  # 64 x 10 weights
    b3: Tensor   # 10 biases
  }

# Adam state for all parameters
type MNISTAdamState:
  State {
    m_w1: Tensor, v_w1: Tensor,
    m_b1: Tensor, v_b1: Tensor,
    m_w2: Tensor, v_w2: Tensor,
    m_b2: Tensor, v_b2: Tensor,
    m_w3: Tensor, v_w3: Tensor,
    m_b3: Tensor, v_b3: Tensor,
    t: u24
  }

# ============================================================
# Weight Initialization
# ============================================================

# Xavier/Glorot initialization for better convergence
def init_weights(rows: u24, cols: u24, seed: u24) -> Tensor:
  # Scale factor: sqrt(2 / (fan_in + fan_out))
  total = rows + cols
  scale = sqrt_init(2.0 / total)

  total_weights = rows * cols
  bend i=0, s=seed:
    when i < total_weights:
      # Random value in [-1, 1] scaled by Xavier factor
      rand = (lcg_init(s) - 0.5) * 2.0 * scale
      result = Tensor/Node {
        left: Tensor/Leaf { val: rand },
        right: fork(i + 1, s + 1)
      }
    else:
      result = Tensor/Nil
  return result

def lcg_init(seed: u24) -> f24:
  a = 1103515245
  c = 12345
  m = 2147483648
  next_val = (a * seed + c) % m
  return next_val / m

def sqrt_init(x: f24) -> f24:
  if x < 0.0001:
    return 0.01
  else:
    guess = x * 0.5
    guess = 0.5 * (guess + x / guess)
    guess = 0.5 * (guess + x / guess)
    return guess

# Initialize biases to small positive values
def init_biases(size: u24) -> Tensor:
  bend i=0:
    when i < size:
      result = Tensor/Node {
        left: Tensor/Leaf { val: 0.01 },
        right: fork(i + 1)
      }
    else:
      result = Tensor/Nil
  return result

# Create initialized network
def create_network(seed: u24) -> MNISTNetwork:
  return MNISTNetwork/Net {
    w1: init_weights(784, 128, seed),
    b1: init_biases(128),
    w2: init_weights(128, 64, seed + 100000),
    b2: init_biases(64),
    w3: init_weights(64, 10, seed + 200000),
    b3: init_biases(10)
  }

# Initialize Adam state
def create_adam_state(net: MNISTNetwork) -> MNISTAdamState:
  match net:
    case MNISTNetwork/Net:
      return MNISTAdamState/State {
        m_w1: zeros_like(net.w1), v_w1: zeros_like(net.w1),
        m_b1: zeros_like(net.b1), v_b1: zeros_like(net.b1),
        m_w2: zeros_like(net.w2), v_w2: zeros_like(net.w2),
        m_b2: zeros_like(net.b2), v_b2: zeros_like(net.b2),
        m_w3: zeros_like(net.w3), v_w3: zeros_like(net.w3),
        m_b3: zeros_like(net.b3), v_b3: zeros_like(net.b3),
        t: 0
      }

def zeros_like(t: Tensor) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: zeros_like(t.left),
        right: zeros_like(t.right)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: 0.0 }
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# Forward Pass
# ============================================================

# Forward pass through entire network
def forward(net: MNISTNetwork, input: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
  # Returns (z1, a1, z2, a2, output)
  # We need intermediate values for backprop
  match net:
    case MNISTNetwork/Net:
      # Layer 1: input -> hidden1
      z1 = dense_forward(input, net.w1, net.b1, 784, 128)
      a1 = relu_tensor(z1)

      # Layer 2: hidden1 -> hidden2
      z2 = dense_forward(a1, net.w2, net.b2, 128, 64)
      a2 = relu_tensor(z2)

      # Layer 3: hidden2 -> output (with softmax)
      z3 = dense_forward(a2, net.w3, net.b3, 64, 10)
      output = softmax_tensor(z3)

      return (a1, a2, z3, output)

# Dense layer forward: output = input @ weights + biases
def dense_forward(input: Tensor, weights: Tensor, biases: Tensor, in_size: u24, out_size: u24) -> Tensor:
  # Simplified matrix-vector multiplication
  bend i=0:
    when i < out_size:
      # Compute dot product for neuron i
      dot = compute_dot_product(input, weights, i, in_size)
      bias = get_bias(biases, i)
      result = Tensor/Node {
        left: Tensor/Leaf { val: dot + bias },
        right: fork(i + 1)
      }
    else:
      result = Tensor/Nil
  return result

# Compute dot product for single output neuron
def compute_dot_product(input: Tensor, weights: Tensor, out_idx: u24, in_size: u24) -> f24:
  # Sum over input dimension
  fold input with weights:
    case Tensor/Node:
      match weights:
        case Tensor/Node:
          return input.left(weights.left) + input.right(weights.right)
        case _:
          return 0.0
    case Tensor/Leaf:
      # Get corresponding weight
      w = get_weight_at(weights, out_idx)
      return input.val * w
    case Tensor/Nil:
      return 0.0

def get_weight_at(weights: Tensor, idx: u24) -> f24:
  return tensor_get(weights, idx, 0)

def tensor_get(t: Tensor, target: u24, current: u24) -> f24:
  match t:
    case Tensor/Node:
      left_count = tensor_count(t.left)
      if target < current + left_count:
        return tensor_get(t.left, target, current)
      else:
        return tensor_get(t.right, target, current + left_count)
    case Tensor/Leaf:
      if current == target:
        return t.val
      else:
        return 0.0
    case Tensor/Nil:
      return 0.0

def tensor_count(t: Tensor) -> u24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return 1
    case Tensor/Nil:
      return 0

def get_bias(biases: Tensor, idx: u24) -> f24:
  return tensor_get(biases, idx, 0)

# ReLU activation
def relu_tensor(t: Tensor) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: relu_tensor(t.left),
        right: relu_tensor(t.right)
      }
    case Tensor/Leaf:
      if t.val > 0.0:
        return Tensor/Leaf { val: t.val }
      else:
        return Tensor/Leaf { val: 0.0 }
    case Tensor/Nil:
      return Tensor/Nil

# Softmax activation (numerically stable)
def softmax_tensor(t: Tensor) -> Tensor:
  max_val = tensor_max(t)
  exp_shifted = tensor_exp_shifted(t, max_val)
  sum_exp = tensor_sum(exp_shifted)
  return tensor_divide(exp_shifted, sum_exp)

def tensor_max(t: Tensor) -> f24:
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

def tensor_exp_shifted(t: Tensor, max_val: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tensor_exp_shifted(t.left, max_val),
        right: tensor_exp_shifted(t.right, max_val)
      }
    case Tensor/Leaf:
      x = t.val - max_val
      return Tensor/Leaf { val: exp_approx(x) }
    case Tensor/Nil:
      return Tensor/Nil

def exp_approx(x: f24) -> f24:
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

def tensor_sum(t: Tensor) -> f24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return t.val
    case Tensor/Nil:
      return 0.0

def tensor_divide(t: Tensor, divisor: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tensor_divide(t.left, divisor),
        right: tensor_divide(t.right, divisor)
      }
    case Tensor/Leaf:
      if divisor > 0.0001:
        return Tensor/Leaf { val: t.val / divisor }
      else:
        return Tensor/Leaf { val: 0.0 }
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# Loss Function
# ============================================================

# Cross-entropy loss: -sum(target * log(pred))
def cross_entropy_loss(pred: Tensor, target: Tensor) -> f24:
  ce = tensor_cross_entropy(pred, target)
  return tensor_sum(ce)

def tensor_cross_entropy(pred: Tensor, target: Tensor) -> Tensor:
  match pred:
    case Tensor/Node:
      match target:
        case Tensor/Node:
          return Tensor/Node {
            left: tensor_cross_entropy(pred.left, target.left),
            right: tensor_cross_entropy(pred.right, target.right)
          }
        case _:
          return Tensor/Nil
    case Tensor/Leaf:
      match target:
        case Tensor/Leaf:
          # -target * log(pred)
          return Tensor/Leaf { val: -target.val * safe_log(pred.val) }
        case _:
          return Tensor/Nil
    case Tensor/Nil:
      return Tensor/Nil

def safe_log(x: f24) -> f24:
  if x < 0.0001:
    return -9.21  # log(0.0001)
  else:
    if x > 0.9999:
      return 0.0
    else:
      t = x - 1.0
      return t - t * t * 0.5 + t * t * t * 0.333

# ============================================================
# Backward Pass
# ============================================================

# Compute gradients through backpropagation
def backward(
  net: MNISTNetwork,
  input: Tensor,
  target: Tensor,
  a1: Tensor,
  a2: Tensor,
  output: Tensor
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
  # Returns (grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3)
  match net:
    case MNISTNetwork/Net:
      # Output layer gradient (softmax + cross-entropy)
      # dL/dz3 = output - target
      delta3 = tensor_subtract(output, target)

      # Gradients for layer 3
      grad_w3 = outer_product(a2, delta3)
      grad_b3 = delta3

      # Backprop to layer 2
      # delta2 = (W3^T @ delta3) * relu'(z2)
      delta2_pre = matmul_transpose(net.w3, delta3, 64, 10)
      delta2 = tensor_mul_relu_deriv(delta2_pre, a2)

      # Gradients for layer 2
      grad_w2 = outer_product(a1, delta2)
      grad_b2 = delta2

      # Backprop to layer 1
      delta1_pre = matmul_transpose(net.w2, delta2, 128, 64)
      delta1 = tensor_mul_relu_deriv(delta1_pre, a1)

      # Gradients for layer 1
      grad_w1 = outer_product(input, delta1)
      grad_b1 = delta1

      return (grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3)

def tensor_subtract(a: Tensor, b: Tensor) -> Tensor:
  match a:
    case Tensor/Node:
      match b:
        case Tensor/Node:
          return Tensor/Node {
            left: tensor_subtract(a.left, b.left),
            right: tensor_subtract(a.right, b.right)
          }
        case _:
          return a
    case Tensor/Leaf:
      match b:
        case Tensor/Leaf:
          return Tensor/Leaf { val: a.val - b.val }
        case _:
          return a
    case Tensor/Nil:
      return b

# Outer product for weight gradients
def outer_product(a: Tensor, b: Tensor) -> Tensor:
  # Creates matrix where out[i,j] = a[i] * b[j]
  match a:
    case Tensor/Node:
      return Tensor/Node {
        left: outer_product(a.left, b),
        right: outer_product(a.right, b)
      }
    case Tensor/Leaf:
      return tensor_scale(b, a.val)
    case Tensor/Nil:
      return Tensor/Nil

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

# Matrix transpose multiply: W^T @ delta
def matmul_transpose(weights: Tensor, delta: Tensor, in_size: u24, out_size: u24) -> Tensor:
  # Result has in_size elements
  bend i=0:
    when i < in_size:
      # Sum over output dimension: sum_j(W[i,j] * delta[j])
      val = compute_transpose_dot(weights, delta, i, in_size, out_size)
      result = Tensor/Node {
        left: Tensor/Leaf { val: val },
        right: fork(i + 1)
      }
    else:
      result = Tensor/Nil
  return result

def compute_transpose_dot(weights: Tensor, delta: Tensor, row: u24, in_size: u24, out_size: u24) -> f24:
  # This is a simplified version - full implementation would iterate over columns
  fold delta with weights:
    case Tensor/Node:
      match weights:
        case Tensor/Node:
          return delta.left(weights.left) + delta.right(weights.right)
        case _:
          return 0.0
    case Tensor/Leaf:
      match weights:
        case Tensor/Leaf:
          return delta.val * weights.val
        case _:
          return 0.0
    case Tensor/Nil:
      return 0.0

# Element-wise multiply with ReLU derivative
def tensor_mul_relu_deriv(delta: Tensor, activation: Tensor) -> Tensor:
  match delta:
    case Tensor/Node:
      match activation:
        case Tensor/Node:
          return Tensor/Node {
            left: tensor_mul_relu_deriv(delta.left, activation.left),
            right: tensor_mul_relu_deriv(delta.right, activation.right)
          }
        case _:
          return Tensor/Nil
    case Tensor/Leaf:
      match activation:
        case Tensor/Leaf:
          # ReLU derivative: 1 if a > 0, else 0
          if activation.val > 0.0:
            return Tensor/Leaf { val: delta.val }
          else:
            return Tensor/Leaf { val: 0.0 }
        case _:
          return Tensor/Nil
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# Adam Optimizer Update
# ============================================================

def adam_step(
  net: MNISTNetwork,
  state: MNISTAdamState,
  grad_w1: Tensor, grad_b1: Tensor,
  grad_w2: Tensor, grad_b2: Tensor,
  grad_w3: Tensor, grad_b3: Tensor,
  lr: f24,
  beta1: f24,
  beta2: f24,
  epsilon: f24
) -> (MNISTNetwork, MNISTAdamState):
  match net:
    case MNISTNetwork/Net:
      match state:
        case MNISTAdamState/State:
          new_t = state.t + 1

          # Update moments and weights for each layer
          (new_w1, new_m_w1, new_v_w1) = adam_update_param(
            net.w1, grad_w1, state.m_w1, state.v_w1,
            new_t, lr, beta1, beta2, epsilon
          )
          (new_b1, new_m_b1, new_v_b1) = adam_update_param(
            net.b1, grad_b1, state.m_b1, state.v_b1,
            new_t, lr, beta1, beta2, epsilon
          )
          (new_w2, new_m_w2, new_v_w2) = adam_update_param(
            net.w2, grad_w2, state.m_w2, state.v_w2,
            new_t, lr, beta1, beta2, epsilon
          )
          (new_b2, new_m_b2, new_v_b2) = adam_update_param(
            net.b2, grad_b2, state.m_b2, state.v_b2,
            new_t, lr, beta1, beta2, epsilon
          )
          (new_w3, new_m_w3, new_v_w3) = adam_update_param(
            net.w3, grad_w3, state.m_w3, state.v_w3,
            new_t, lr, beta1, beta2, epsilon
          )
          (new_b3, new_m_b3, new_v_b3) = adam_update_param(
            net.b3, grad_b3, state.m_b3, state.v_b3,
            new_t, lr, beta1, beta2, epsilon
          )

          new_net = MNISTNetwork/Net {
            w1: new_w1, b1: new_b1,
            w2: new_w2, b2: new_b2,
            w3: new_w3, b3: new_b3
          }

          new_state = MNISTAdamState/State {
            m_w1: new_m_w1, v_w1: new_v_w1,
            m_b1: new_m_b1, v_b1: new_v_b1,
            m_w2: new_m_w2, v_w2: new_v_w2,
            m_b2: new_m_b2, v_b2: new_v_b2,
            m_w3: new_m_w3, v_w3: new_v_w3,
            m_b3: new_m_b3, v_b3: new_v_b3,
            t: new_t
          }

          return (new_net, new_state)

def adam_update_param(
  param: Tensor,
  grad: Tensor,
  m: Tensor,
  v: Tensor,
  t: u24,
  lr: f24,
  beta1: f24,
  beta2: f24,
  eps: f24
) -> (Tensor, Tensor, Tensor):
  # Update moments
  new_m = update_moment(m, grad, beta1)
  new_v = update_variance(v, grad, beta2)

  # Bias correction
  beta1_t = pow_approx_int(beta1, t)
  beta2_t = pow_approx_int(beta2, t)
  m_corr = 1.0 / (1.0 - beta1_t)
  v_corr = 1.0 / (1.0 - beta2_t)

  # Update parameters
  new_param = apply_adam(param, new_m, new_v, lr, m_corr, v_corr, eps)

  return (new_param, new_m, new_v)

def update_moment(m: Tensor, grad: Tensor, beta: f24) -> Tensor:
  match m:
    case Tensor/Node:
      match grad:
        case Tensor/Node:
          return Tensor/Node {
            left: update_moment(m.left, grad.left, beta),
            right: update_moment(m.right, grad.right, beta)
          }
        case _:
          return m
    case Tensor/Leaf:
      match grad:
        case Tensor/Leaf:
          return Tensor/Leaf { val: beta * m.val + (1.0 - beta) * grad.val }
        case _:
          return m
    case Tensor/Nil:
      return Tensor/Nil

def update_variance(v: Tensor, grad: Tensor, beta: f24) -> Tensor:
  match v:
    case Tensor/Node:
      match grad:
        case Tensor/Node:
          return Tensor/Node {
            left: update_variance(v.left, grad.left, beta),
            right: update_variance(v.right, grad.right, beta)
          }
        case _:
          return v
    case Tensor/Leaf:
      match grad:
        case Tensor/Leaf:
          return Tensor/Leaf { val: beta * v.val + (1.0 - beta) * grad.val * grad.val }
        case _:
          return v
    case Tensor/Nil:
      return Tensor/Nil

def pow_approx_int(base: f24, exp: u24) -> f24:
  bend i=0, result=1.0:
    when i < exp:
      fork(i + 1, result * base)
    else:
      result = result
  return result

def apply_adam(param: Tensor, m: Tensor, v: Tensor, lr: f24, m_corr: f24, v_corr: f24, eps: f24) -> Tensor:
  match param:
    case Tensor/Node:
      match m:
        case Tensor/Node:
          match v:
            case Tensor/Node:
              return Tensor/Node {
                left: apply_adam(param.left, m.left, v.left, lr, m_corr, v_corr, eps),
                right: apply_adam(param.right, m.right, v.right, lr, m_corr, v_corr, eps)
              }
            case _:
              return param
        case _:
          return param
    case Tensor/Leaf:
      match m:
        case Tensor/Leaf:
          match v:
            case Tensor/Leaf:
              m_hat = m.val * m_corr
              v_hat = v.val * v_corr
              return Tensor/Leaf { val: param.val - lr * m_hat / (sqrt_init(v_hat) + eps) }
            case _:
              return param
        case _:
          return param
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# Training Step
# ============================================================

def train_step(
  net: MNISTNetwork,
  state: MNISTAdamState,
  input: Tensor,
  target: Tensor,
  lr: f24
) -> (MNISTNetwork, MNISTAdamState, f24):
  # Forward pass
  (a1, a2, z3, output) = forward(net, input)

  # Compute loss
  loss = cross_entropy_loss(output, target)

  # Backward pass
  (grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3) =
    backward(net, input, target, a1, a2, output)

  # Adam update
  (new_net, new_state) = adam_step(
    net, state,
    grad_w1, grad_b1,
    grad_w2, grad_b2,
    grad_w3, grad_b3,
    lr, 0.9, 0.999, 0.00000001
  )

  return (new_net, new_state, loss)

# ============================================================
# Evaluation
# ============================================================

def predict(net: MNISTNetwork, input: Tensor) -> u24:
  (_, _, _, output) = forward(net, input)
  return argmax(output)

def argmax(t: Tensor) -> u24:
  return find_max_idx(t, 0, -999999.0, 0)

def find_max_idx(t: Tensor, idx: u24, max_val: f24, max_idx: u24) -> u24:
  match t:
    case Tensor/Node:
      left_result = find_max_idx(t.left, idx, max_val, max_idx)
      left_count = tensor_count(t.left)
      left_max = get_max_in_subtree(t.left, max_val)
      return find_max_idx(t.right, idx + left_count, left_max, left_result)
    case Tensor/Leaf:
      if t.val > max_val:
        return idx
      else:
        return max_idx
    case Tensor/Nil:
      return max_idx

def get_max_in_subtree(t: Tensor, current: f24) -> f24:
  fold t:
    case Tensor/Node:
      left = t.left
      right = t.right
      if left > right:
        if left > current:
          return left
        else:
          return current
      else:
        if right > current:
          return right
        else:
          return current
    case Tensor/Leaf:
      if t.val > current:
        return t.val
      else:
        return current
    case Tensor/Nil:
      return current

# ============================================================
# Test Data Generation
# ============================================================

# Generate test image for a digit
def generate_digit_image(digit: u24, seed: u24) -> Tensor:
  bend i=0, s=seed:
    when i < 784:
      row = i / 28
      col = i % 28
      pixel = generate_pixel(digit, row, col, s)
      result = Tensor/Node {
        left: Tensor/Leaf { val: pixel / 255.0 },
        right: fork(i + 1, s + 1)
      }
    else:
      result = Tensor/Nil
  return result

def generate_pixel(digit: u24, row: u24, col: u24, seed: u24) -> f24:
  center_row = 14
  center_col = 14
  dr = if row > center_row { row - center_row } else { center_row - row }
  dc = if col > center_col { col - center_col } else { center_col - col }
  dist = dr + dc

  if dist < 5 + digit:
    return 200.0 + (lcg_init(seed) - 0.5) * 50.0
  else:
    if dist < 10 + digit:
      return 50.0 + (lcg_init(seed + 1) - 0.5) * 30.0
    else:
      return lcg_init(seed + 2) * 20.0

# One-hot encode label
def one_hot_label(label: u24) -> Tensor:
  bend i=0:
    when i < 10:
      if i == label:
        result = Tensor/Node {
          left: Tensor/Leaf { val: 1.0 },
          right: fork(i + 1)
        }
      else:
        result = Tensor/Node {
          left: Tensor/Leaf { val: 0.0 },
          right: fork(i + 1)
        }
    else:
      result = Tensor/Nil
  return result

# ============================================================
# Main Training Loop
# ============================================================

def train_mnist(epochs: u24, samples_per_epoch: u24, learning_rate: f24) -> f24:
  # Initialize network and optimizer
  net = create_network(42)
  state = create_adam_state(net)

  # Training loop
  (final_net, final_state, total_loss) = train_epochs(net, state, epochs, samples_per_epoch, learning_rate, 0, 0.0)

  # Return average loss
  total_samples = epochs * samples_per_epoch
  if total_samples > 0:
    return total_loss / total_samples
  else:
    return 0.0

def train_epochs(
  net: MNISTNetwork,
  state: MNISTAdamState,
  epochs: u24,
  samples: u24,
  lr: f24,
  current_epoch: u24,
  acc_loss: f24
) -> (MNISTNetwork, MNISTAdamState, f24):
  if current_epoch >= epochs:
    return (net, state, acc_loss)
  else:
    # Train one epoch
    (new_net, new_state, epoch_loss) = train_one_epoch(net, state, samples, lr, current_epoch * 1000)

    # Continue to next epoch
    return train_epochs(new_net, new_state, epochs, samples, lr, current_epoch + 1, acc_loss + epoch_loss)

def train_one_epoch(
  net: MNISTNetwork,
  state: MNISTAdamState,
  samples: u24,
  lr: f24,
  seed: u24
) -> (MNISTNetwork, MNISTAdamState, f24):
  return train_samples(net, state, samples, lr, seed, 0, 0.0)

def train_samples(
  net: MNISTNetwork,
  state: MNISTAdamState,
  total_samples: u24,
  lr: f24,
  seed: u24,
  current: u24,
  acc_loss: f24
) -> (MNISTNetwork, MNISTAdamState, f24):
  if current >= total_samples:
    return (net, state, acc_loss)
  else:
    # Generate training sample
    digit = current % 10
    input = generate_digit_image(digit, seed + current * 100)
    target = one_hot_label(digit)

    # Train step
    (new_net, new_state, loss) = train_step(net, state, input, target, lr)

    # Continue
    return train_samples(new_net, new_state, total_samples, lr, seed, current + 1, acc_loss + loss)

# ============================================================
# Entry Point
# ============================================================

def main() -> f24:
  # Train for 5 epochs with 100 samples per epoch
  # Learning rate: 0.001
  avg_loss = train_mnist(5, 100, 0.001)
  return avg_loss
