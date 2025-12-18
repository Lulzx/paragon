# RL Training Example with Interaction Net Credit Assignment
# Full end-to-end training on a simple environment
#
# This demonstrates:
# 1. Policy network initialization
# 2. Episode collection
# 3. Credit assignment via interaction nets
# 4. Policy gradient updates
# 5. Training loop with metrics

# ============================================================
# Core Types (from Paragon)
# ============================================================

type Tensor:
  Node { ~left: Tensor, ~right: Tensor }
  Leaf { val: f24 }
  Nil

type Vector:
  Vec { len: u24, data: Tensor }
  Empty

# ============================================================
# Policy Network
# ============================================================

type Policy:
  PolicyNet {
    weights: Tensor,
    biases: Tensor,
    input_size: u24,
    hidden_size: u24,
    output_size: u24
  }

# ============================================================
# Environment
# ============================================================

type Environment:
  Env {
    state: Tensor,
    done: u24,
    step_count: u24,
    max_steps: u24
  }

# ============================================================
# Trajectory
# ============================================================

type TrajectoryStep:
  Step {
    time: u24,
    state: Tensor,
    action: u24,
    action_logits: Tensor,
    log_prob: f24,
    next_state: Tensor,
    reward: f24,
    done: u24
  }

type Trajectory:
  TCons { head: TrajectoryStep, ~tail: Trajectory }
  TNil

# ============================================================
# Credit Map
# ============================================================

type CreditMap:
  CMapCons { time: u24, credit: f24, ~tail: CreditMap }
  CMapNil

# ============================================================
# Training State
# ============================================================

type TrainState:
  State {
    policy: Policy,
    episode: u24,
    total_reward: f24,
    avg_reward: f24
  }

# ============================================================
# Utility Functions
# ============================================================

def lcg_rand_rl(seed: u24) -> f24:
  a = 1103515245
  c = 12345
  m = 2147483648
  next = (a * seed + c) % m
  return next / m

# ============================================================
# Policy Initialization
# ============================================================

def policy_init_rl(input_size: u24, hidden_size: u24, output_size: u24, seed: u24) -> Policy:
  weights = init_weights_rl(input_size * hidden_size + hidden_size * output_size, seed)
  biases = init_biases_rl(hidden_size + output_size)
  return Policy/PolicyNet {
    weights: weights,
    biases: biases,
    input_size: input_size,
    hidden_size: hidden_size,
    output_size: output_size
  }

def init_weights_rl(total: u24, seed: u24) -> Tensor:
  scale = 0.1
  bend i=0, s=seed:
    when i < total:
      w = lcg_rand_rl(s)
      result = Tensor/Node {
        left: Tensor/Leaf { val: (w - 0.5) * scale },
        right: fork(i + 1, s + 1)
      }
    else:
      result = Tensor/Nil
  return result

def init_biases_rl(total: u24) -> Tensor:
  bend i=0:
    when i < total:
      result = Tensor/Node {
        left: Tensor/Leaf { val: 0.0 },
        right: fork(i + 1)
      }
    else:
      result = Tensor/Nil
  return result

# ============================================================
# Policy Forward Pass
# ============================================================

def policy_forward_rl(policy: Policy, state: Tensor) -> Tensor:
  match policy:
    case Policy/PolicyNet:
      hidden = tree_weighted_mul(policy.weights, state)
      hidden_relu = tree_relu_rl(hidden)
      return tree_weighted_mul(policy.weights, hidden_relu)

def tree_weighted_mul(weights: Tensor, input: Tensor) -> Tensor:
  match weights:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_weighted_mul(weights.left, input),
        right: tree_weighted_mul(weights.right, input)
      }
    case Tensor/Leaf:
      match input:
        case Tensor/Leaf:
          return Tensor/Leaf { val: weights.val * input.val }
        case _:
          return Tensor/Leaf { val: 0.0 }
    case Tensor/Nil:
      return Tensor/Nil

def tree_relu_rl(t: Tensor) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_relu_rl(t.left),
        right: tree_relu_rl(t.right)
      }
    case Tensor/Leaf:
      if t.val > 0.0:
        return Tensor/Leaf { val: t.val }
      else:
        return Tensor/Leaf { val: 0.0 }
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# Action Sampling
# ============================================================

def policy_sample_rl(policy: Policy, state: Tensor, seed: u24) -> (u24, f24, Tensor):
  logits = policy_forward_rl(policy, state)
  probs = softmax_rl(logits)
  u = lcg_rand_rl(seed)
  action = sample_categorical_rl(probs, u)
  log_prob = get_log_prob_rl(probs, action)
  return (action, log_prob, logits)

def softmax_rl(logits: Tensor) -> Tensor:
  max_val = tree_max_rl(logits)
  exp_shifted = tree_exp_shifted_rl(logits, max_val)
  sum_exp = tree_sum_rl(exp_shifted)
  return tree_divide_rl(exp_shifted, sum_exp)

def tree_max_rl(t: Tensor) -> f24:
  fold t:
    case Tensor/Node:
      left = t.left
      right = t.right
      if left > right:
        return left
      else:
        return right
    case Tensor/Leaf:
      return t.val
    case Tensor/Nil:
      return -999999.0

def tree_exp_shifted_rl(t: Tensor, max_val: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_exp_shifted_rl(t.left, max_val),
        right: tree_exp_shifted_rl(t.right, max_val)
      }
    case Tensor/Leaf:
      x = t.val - max_val
      return Tensor/Leaf { val: exp_approx_rl(x) }
    case Tensor/Nil:
      return Tensor/Nil

def exp_approx_rl(x: f24) -> f24:
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

def tree_sum_rl(t: Tensor) -> f24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return t.val
    case Tensor/Nil:
      return 0.0

def tree_divide_rl(t: Tensor, divisor: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_divide_rl(t.left, divisor),
        right: tree_divide_rl(t.right, divisor)
      }
    case Tensor/Leaf:
      if divisor > 0.0001:
        return Tensor/Leaf { val: t.val / divisor }
      else:
        return Tensor/Leaf { val: 0.0 }
    case Tensor/Nil:
      return Tensor/Nil

def sample_categorical_rl(probs: Tensor, u: f24) -> u24:
  return sample_cat_rec(probs, u, 0.0, 0)

def sample_cat_rec(probs: Tensor, u: f24, cumsum: f24, idx: u24) -> u24:
  match probs:
    case Tensor/Node:
      left_result = sample_cat_rec(probs.left, u, cumsum, idx)
      if left_result != 999999:
        return left_result
      else:
        left_sum = tree_sum_rl(probs.left)
        left_count = tree_count_rl(probs.left)
        return sample_cat_rec(probs.right, u, cumsum + left_sum, idx + left_count)
    case Tensor/Leaf:
      new_cumsum = cumsum + probs.val
      if u < new_cumsum:
        return idx
      else:
        return 999999
    case Tensor/Nil:
      return 0

def tree_count_rl(t: Tensor) -> u24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return 1
    case Tensor/Nil:
      return 0

def get_log_prob_rl(probs: Tensor, action: u24) -> f24:
  prob = tree_get_at_rl(probs, action, 0)
  if prob > 0.0001:
    return log_approx_rl(prob)
  else:
    return -10.0

def tree_get_at_rl(t: Tensor, target: u24, current: u24) -> f24:
  match t:
    case Tensor/Node:
      left_count = tree_count_rl(t.left)
      if target < current + left_count:
        return tree_get_at_rl(t.left, target, current)
      else:
        return tree_get_at_rl(t.right, target, current + left_count)
    case Tensor/Leaf:
      if current == target:
        return t.val
      else:
        return 0.0
    case Tensor/Nil:
      return 0.0

def log_approx_rl(x: f24) -> f24:
  if x > 2.0:
    return 0.693 + log_approx_rl(x * 0.5)
  else:
    if x < 0.5:
      return -0.693 + log_approx_rl(x * 2.0)
    else:
      y = x - 1.0
      return y - y * y * 0.5 + y * y * y * 0.333

# ============================================================
# Environment Functions
# ============================================================

def env_reset_rl(max_steps: u24, seed: u24) -> Environment:
  return Environment/Env {
    state: init_state_rl(seed),
    done: 0,
    step_count: 0,
    max_steps: max_steps
  }

def init_state_rl(seed: u24) -> Tensor:
  bend i=0, s=seed:
    when i < 8:
      val = lcg_rand_rl(s) - 0.5
      result = Tensor/Node {
        left: Tensor/Leaf { val: val },
        right: fork(i + 1, s + 1)
      }
    else:
      result = Tensor/Nil
  return result

def env_step_rl(env: Environment, action: u24) -> (Environment, f24):
  match env:
    case Environment/Env:
      new_state = apply_action_rl(env.state, action)
      reward = compute_reward_rl(new_state)
      new_step = env.step_count + 1
      is_done = if new_step >= env.max_steps: 1 else: 0
      new_env = Environment/Env {
        state: new_state,
        done: is_done,
        step_count: new_step,
        max_steps: env.max_steps
      }
      return (new_env, reward)

def apply_action_rl(state: Tensor, action: u24) -> Tensor:
  delta = (action + 1) * 0.1 - 0.2
  return tree_add_scalar_rl(state, delta)

def tree_add_scalar_rl(t: Tensor, s: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_add_scalar_rl(t.left, s),
        right: tree_add_scalar_rl(t.right, s)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: t.val + s }
    case Tensor/Nil:
      return Tensor/Nil

def compute_reward_rl(state: Tensor) -> f24:
  return tree_sum_rl(state)

def env_is_done_rl(env: Environment) -> u24:
  match env:
    case Environment/Env:
      return env.done

def env_get_state_rl(env: Environment) -> Tensor:
  match env:
    case Environment/Env:
      return env.state

# ============================================================
# Trajectory Functions
# ============================================================

def trajectory_new_rl() -> Trajectory:
  return Trajectory/TNil

def trajectory_add_rl(traj: Trajectory, step: TrajectoryStep) -> Trajectory:
  return Trajectory/TCons { head: step, tail: traj }

def trajectory_reverse_rl(traj: Trajectory) -> Trajectory:
  return traj_rev_rl(traj, Trajectory/TNil)

def traj_rev_rl(traj: Trajectory, acc: Trajectory) -> Trajectory:
  match traj:
    case Trajectory/TCons:
      return traj_rev_rl(traj.tail, Trajectory/TCons { head: traj.head, tail: acc })
    case Trajectory/TNil:
      return acc

def trajectory_total_reward_rl(traj: Trajectory) -> f24:
  match traj:
    case Trajectory/TCons:
      match traj.head:
        case TrajectoryStep/Step:
          return traj.head.reward + trajectory_total_reward_rl(traj.tail)
    case Trajectory/TNil:
      return 0.0

def trajectory_length_rl(traj: Trajectory) -> u24:
  match traj:
    case Trajectory/TCons:
      return 1 + trajectory_length_rl(traj.tail)
    case Trajectory/TNil:
      return 0

# ============================================================
# Credit Assignment
# ============================================================

def assign_credit_rl(traj: Trajectory, gamma: f24) -> CreditMap:
  total_reward = trajectory_total_reward_rl(traj)
  length = trajectory_length_rl(traj)
  return assign_credit_rec_rl(traj, 0, length, total_reward, gamma, CreditMap/CMapNil)

def assign_credit_rec_rl(traj: Trajectory, t: u24, length: u24, total_reward: f24, gamma: f24, acc: CreditMap) -> CreditMap:
  match traj:
    case Trajectory/TCons:
      match traj.head:
        case TrajectoryStep/Step:
          distance = length - t
          discount = pow_gamma_rl(gamma, distance)
          credit = total_reward * discount
          new_acc = CreditMap/CMapCons { time: traj.head.time, credit: credit, tail: acc }
          return assign_credit_rec_rl(traj.tail, t + 1, length, total_reward, gamma, new_acc)
    case Trajectory/TNil:
      return acc

def pow_gamma_rl(base: f24, exp: u24) -> f24:
  if exp == 0:
    return 1.0
  else:
    if exp == 1:
      return base
    else:
      half = pow_gamma_rl(base, exp / 2)
      if exp % 2 == 0:
        return half * half
      else:
        return half * half * base

def credit_map_get_rl(map: CreditMap, time: u24) -> f24:
  match map:
    case CreditMap/CMapCons:
      if map.time == time:
        return map.credit
      else:
        return credit_map_get_rl(map.tail, time)
    case CreditMap/CMapNil:
      return 0.0

# ============================================================
# Policy Gradient Update
# ============================================================

def policy_update_rl(policy: Policy, traj: Trajectory, credits: CreditMap, lr: f24) -> Policy:
  match policy:
    case Policy/PolicyNet:
      grad = compute_gradient_rl(traj, credits)
      new_weights = update_weights_rl(policy.weights, grad, lr)
      return Policy/PolicyNet {
        weights: new_weights,
        biases: policy.biases,
        input_size: policy.input_size,
        hidden_size: policy.hidden_size,
        output_size: policy.output_size
      }

def compute_gradient_rl(traj: Trajectory, credits: CreditMap) -> Tensor:
  match traj:
    case Trajectory/TCons:
      match traj.head:
        case TrajectoryStep/Step:
          advantage = credit_map_get_rl(credits, traj.head.time)
          step_grad = 0.0 - traj.head.log_prob * advantage
          rest_grad = compute_gradient_rl(traj.tail, credits)
          return tree_add_scalar_rl(rest_grad, step_grad)
    case Trajectory/TNil:
      return Tensor/Leaf { val: 0.0 }

def update_weights_rl(weights: Tensor, grad: Tensor, lr: f24) -> Tensor:
  match weights:
    case Tensor/Node:
      match grad:
        case Tensor/Node:
          return Tensor/Node {
            left: update_weights_rl(weights.left, grad.left, lr),
            right: update_weights_rl(weights.right, grad.right, lr)
          }
        case Tensor/Leaf:
          return Tensor/Node {
            left: update_weights_rl(weights.left, grad, lr),
            right: update_weights_rl(weights.right, grad, lr)
          }
        case Tensor/Nil:
          return weights
    case Tensor/Leaf:
      match grad:
        case Tensor/Leaf:
          return Tensor/Leaf { val: weights.val - lr * grad.val }
        case _:
          return weights
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# Episode Collection
# ============================================================

def collect_episode_rl(policy: Policy, env: Environment, seed: u24) -> (Trajectory, f24):
  return collect_rec_rl(policy, env, Trajectory/TNil, 0, seed)

def collect_rec_rl(policy: Policy, env: Environment, traj: Trajectory, t: u24, seed: u24) -> (Trajectory, f24):
  if env_is_done_rl(env) == 1:
    return (trajectory_reverse_rl(traj), trajectory_total_reward_rl(traj))
  else:
    state = env_get_state_rl(env)
    sample_result = policy_sample_rl(policy, state, seed + t)
    match sample_result:
      case (action, log_prob, logits):
        step_result = env_step_rl(env, action)
        match step_result:
          case (new_env, reward):
            next_state = env_get_state_rl(new_env)
            is_done = env_is_done_rl(new_env)
            step = TrajectoryStep/Step {
              time: t,
              state: state,
              action: action,
              action_logits: logits,
              log_prob: log_prob,
              next_state: next_state,
              reward: reward,
              done: is_done
            }
            new_traj = trajectory_add_rl(traj, step)
            return collect_rec_rl(policy, new_env, new_traj, t + 1, seed)

# ============================================================
# Training Loop
# ============================================================

def train_rl(policy: Policy, episodes: u24, lr: f24, gamma: f24, max_steps: u24) -> TrainState:
  return train_loop_rl(policy, 0, episodes, lr, gamma, max_steps, 0.0, 0.0)

def train_loop_rl(policy: Policy, ep: u24, total_eps: u24, lr: f24, gamma: f24, max_steps: u24, total_reward: f24, running_avg: f24) -> TrainState:
  if ep >= total_eps:
    return TrainState/State {
      policy: policy,
      episode: ep,
      total_reward: total_reward,
      avg_reward: running_avg
    }
  else:
    env = env_reset_rl(max_steps, ep * 1000)
    episode_result = collect_episode_rl(policy, env, ep * 1000 + 500)
    match episode_result:
      case (traj, ep_reward):
        credits = assign_credit_rl(traj, gamma)
        updated_policy = policy_update_rl(policy, traj, credits, lr)
        alpha = 0.1
        new_avg = alpha * ep_reward + (1.0 - alpha) * running_avg
        return train_loop_rl(updated_policy, ep + 1, total_eps, lr, gamma, max_steps, total_reward + ep_reward, new_avg)

# ============================================================
# Entry Points
# ============================================================

# Run full training
def main() -> f24:
  # Initialize policy: 8 inputs, 16 hidden, 4 actions
  policy = policy_init_rl(8, 16, 4, 42)

  # Train for 100 episodes
  result = train_rl(policy, 100, 0.01, 0.99, 50)

  # Return average reward
  match result:
    case TrainState/State:
      return result.avg_reward

# Quick test with fewer episodes
def quick_test() -> f24:
  policy = policy_init_rl(8, 16, 4, 42)
  result = train_rl(policy, 10, 0.01, 0.99, 20)
  match result:
    case TrainState/State:
      return result.avg_reward

# Single episode test
def single_episode_test() -> f24:
  policy = policy_init_rl(8, 16, 4, 42)
  env = env_reset_rl(20, 42)
  episode_result = collect_episode_rl(policy, env, 100)
  match episode_result:
    case (traj, reward):
      return reward
