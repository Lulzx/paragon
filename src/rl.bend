# Reinforcement Learning Training Loop with Credit Assignment
# Unified substrate: policy, influence model, and credit assignment
# all run on the same interaction net runtime
#
# The training loop:
# 1. Collect trajectory using policy
# 2. Build interaction net from trajectory
# 3. Credit assignment via reduction (parallel!)
# 4. Update policy using policy gradient
# 5. Update influence model (self-supervised)

# ============================================================
# Core Types
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
    weights: Tensor,        # Policy network weights
    biases: Tensor,         # Policy network biases
    input_size: u24,        # State dimension
    hidden_size: u24,       # Hidden layer size
    output_size: u24        # Number of actions
  }

# Create policy network
def policy_init(input_size: u24, hidden_size: u24, output_size: u24, seed: u24) -> Policy:
  return Policy/PolicyNet {
    weights: init_policy_weights(input_size, hidden_size, output_size, seed),
    biases: init_policy_biases(hidden_size, output_size),
    input_size: input_size,
    hidden_size: hidden_size,
    output_size: output_size
  }

def init_policy_weights(input_size: u24, hidden_size: u24, output_size: u24, seed: u24) -> Tensor:
  # Two layer weights: input->hidden and hidden->output
  layer1_size = input_size * hidden_size
  layer2_size = hidden_size * output_size
  total = layer1_size + layer2_size
  scale = 0.1

  bend i=0, s=seed:
    when i < total:
      w = lcg_rand(s)
      result = Tensor/Node {
        left: Tensor/Leaf { val: (w - 0.5) * scale },
        right: fork(i + 1, s + 1)
      }
    else:
      result = Tensor/Nil
  return result

def init_policy_biases(hidden_size: u24, output_size: u24) -> Tensor:
  total = hidden_size + output_size
  bend i=0:
    when i < total:
      result = Tensor/Node {
        left: Tensor/Leaf { val: 0.0 },
        right: fork(i + 1)
      }
    else:
      result = Tensor/Nil
  return result

def lcg_rand(seed: u24) -> f24:
  a = 1103515245
  c = 12345
  m = 2147483648
  next = (a * seed + c) % m
  return next / m

# ============================================================
# Policy Forward Pass
# ============================================================

# Compute action logits from state
def policy_forward(policy: Policy, state: Tensor) -> Tensor:
  match policy:
    case Policy/PolicyNet:
      # Two-layer MLP with ReLU hidden
      hidden = policy_hidden_layer(policy.weights, state, policy.input_size, policy.hidden_size)
      hidden_relu = tree_relu_policy(hidden)
      logits = policy_output_layer(policy.weights, hidden_relu, policy.hidden_size, policy.output_size, policy.input_size)
      return logits

def policy_hidden_layer(weights: Tensor, input: Tensor, input_size: u24, hidden_size: u24) -> Tensor:
  # Simple dense layer simulation
  return tree_weighted_sum_policy(weights, input)

def policy_output_layer(weights: Tensor, hidden: Tensor, hidden_size: u24, output_size: u24, offset: u24) -> Tensor:
  # Output logits from hidden activations
  return tree_weighted_sum_policy(weights, hidden)

def tree_weighted_sum_policy(weights: Tensor, input: Tensor) -> Tensor:
  match weights:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_weighted_sum_policy(weights.left, input),
        right: tree_weighted_sum_policy(weights.right, input)
      }
    case Tensor/Leaf:
      match input:
        case Tensor/Leaf:
          return Tensor/Leaf { val: weights.val * input.val }
        case _:
          return Tensor/Leaf { val: 0.0 }
    case Tensor/Nil:
      return Tensor/Nil

def tree_relu_policy(t: Tensor) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_relu_policy(t.left),
        right: tree_relu_policy(t.right)
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

# Sample action from policy (softmax over logits)
def policy_sample(policy: Policy, state: Tensor, rand_seed: u24) -> (u24, f24, Tensor):
  logits = policy_forward(policy, state)
  probs = softmax_policy(logits)

  # Sample from categorical distribution
  u = lcg_rand(rand_seed)
  action = sample_categorical(probs, u)

  # Compute log probability
  log_prob = log_prob_at(probs, action)

  return (action, log_prob, logits)

def softmax_policy(logits: Tensor) -> Tensor:
  max_val = tree_max_policy(logits)
  exp_shifted = tree_exp_shifted_policy(logits, max_val)
  sum_exp = tree_sum_policy(exp_shifted)
  return tree_divide_policy(exp_shifted, sum_exp)

def tree_max_policy(t: Tensor) -> f24:
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

def tree_exp_shifted_policy(t: Tensor, max_val: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_exp_shifted_policy(t.left, max_val),
        right: tree_exp_shifted_policy(t.right, max_val)
      }
    case Tensor/Leaf:
      x = t.val - max_val
      return Tensor/Leaf { val: exp_approx_policy(x) }
    case Tensor/Nil:
      return Tensor/Nil

def tree_divide_policy(t: Tensor, divisor: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_divide_policy(t.left, divisor),
        right: tree_divide_policy(t.right, divisor)
      }
    case Tensor/Leaf:
      if divisor > 0.0001:
        return Tensor/Leaf { val: t.val / divisor }
      else:
        return Tensor/Leaf { val: 0.0 }
    case Tensor/Nil:
      return Tensor/Nil

def tree_sum_policy(t: Tensor) -> f24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return t.val
    case Tensor/Nil:
      return 0.0

def exp_approx_policy(x: f24) -> f24:
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

# Sample from categorical distribution
def sample_categorical(probs: Tensor, u: f24) -> u24:
  return sample_categorical_rec(probs, u, 0.0, 0)

def sample_categorical_rec(probs: Tensor, u: f24, cumsum: f24, idx: u24) -> u24:
  match probs:
    case Tensor/Node:
      left_result = sample_categorical_rec(probs.left, u, cumsum, idx)
      if left_result != 999999:  # Found in left branch
        return left_result
      else:
        left_sum = tree_sum_policy(probs.left)
        return sample_categorical_rec(probs.right, u, cumsum + left_sum, idx + tree_count_policy(probs.left))
    case Tensor/Leaf:
      new_cumsum = cumsum + probs.val
      if u < new_cumsum:
        return idx
      else:
        return 999999  # Not found yet
    case Tensor/Nil:
      return 0

def tree_count_policy(t: Tensor) -> u24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return 1
    case Tensor/Nil:
      return 0

# Get log probability at specific action index
def log_prob_at(probs: Tensor, action: u24) -> f24:
  prob = tree_get_at(probs, action, 0)
  if prob > 0.0001:
    return log_approx(prob)
  else:
    return -10.0  # Clip for numerical stability

def tree_get_at(t: Tensor, target: u24, current: u24) -> f24:
  match t:
    case Tensor/Node:
      left_count = tree_count_policy(t.left)
      if target < current + left_count:
        return tree_get_at(t.left, target, current)
      else:
        return tree_get_at(t.right, target, current + left_count)
    case Tensor/Leaf:
      if current == target:
        return t.val
      else:
        return 0.0
    case Tensor/Nil:
      return 0.0

def log_approx(x: f24) -> f24:
  # Taylor series for log around 1
  if x > 2.0:
    return 0.693 + log_approx(x * 0.5)
  else:
    if x < 0.5:
      return -0.693 + log_approx(x * 2.0)
    else:
      y = x - 1.0
      y2 = y * y
      y3 = y2 * y
      return y - y2 * 0.5 + y3 * 0.333

# ============================================================
# Environment Interface
# ============================================================

type Environment:
  Env {
    state: Tensor,          # Current state
    done: u24,              # 1 if terminal, 0 otherwise
    step_count: u24,        # Steps taken
    max_steps: u24          # Maximum steps per episode
  }

# Reset environment to initial state
def env_reset(max_steps: u24, seed: u24) -> Environment:
  return Environment/Env {
    state: init_random_state(seed),
    done: 0,
    step_count: 0,
    max_steps: max_steps
  }

def init_random_state(seed: u24) -> Tensor:
  # Simple random initial state
  bend i=0, s=seed:
    when i < 8:  # State size = 8
      val = lcg_rand(s) - 0.5
      result = Tensor/Node {
        left: Tensor/Leaf { val: val },
        right: fork(i + 1, s + 1)
      }
    else:
      result = Tensor/Nil
  return result

# Take action in environment
def env_step(env: Environment, action: u24) -> (Environment, f24):
  match env:
    case Environment/Env:
      # Simple dynamics: state changes based on action
      new_state = apply_action(env.state, action)

      # Simple reward function
      reward = compute_reward(new_state)

      new_step = env.step_count + 1
      is_done = if new_step >= env.max_steps: 1 else: 0

      new_env = Environment/Env {
        state: new_state,
        done: is_done,
        step_count: new_step,
        max_steps: env.max_steps
      }
      return (new_env, reward)

def apply_action(state: Tensor, action: u24) -> Tensor:
  # Simple action effect: shift state values
  delta = (action + 1) * 0.1 - 0.2
  return tree_add_scalar(state, delta)

def tree_add_scalar(t: Tensor, s: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_add_scalar(t.left, s),
        right: tree_add_scalar(t.right, s)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: t.val + s }
    case Tensor/Nil:
      return Tensor/Nil

def compute_reward(state: Tensor) -> f24:
  # Reward = sum of state values (incentivize positive states)
  return tree_sum_policy(state)

def env_is_done(env: Environment) -> u24:
  match env:
    case Environment/Env:
      return env.done

def env_get_state(env: Environment) -> Tensor:
  match env:
    case Environment/Env:
      return env.state

# ============================================================
# Trajectory Types
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

def trajectory_new() -> Trajectory:
  return Trajectory/TNil

def trajectory_add(traj: Trajectory, step: TrajectoryStep) -> Trajectory:
  return Trajectory/TCons { head: step, tail: traj }

def trajectory_reverse(traj: Trajectory) -> Trajectory:
  return traj_rev_acc(traj, Trajectory/TNil)

def traj_rev_acc(traj: Trajectory, acc: Trajectory) -> Trajectory:
  match traj:
    case Trajectory/TCons:
      return traj_rev_acc(traj.tail, Trajectory/TCons { head: traj.head, tail: acc })
    case Trajectory/TNil:
      return acc

def trajectory_length(traj: Trajectory) -> u24:
  match traj:
    case Trajectory/TCons:
      return 1 + trajectory_length(traj.tail)
    case Trajectory/TNil:
      return 0

def trajectory_total_reward(traj: Trajectory) -> f24:
  match traj:
    case Trajectory/TCons:
      match traj.head:
        case TrajectoryStep/Step:
          return traj.head.reward + trajectory_total_reward(traj.tail)
    case Trajectory/TNil:
      return 0.0

# ============================================================
# Credit Map (from reduction.bend)
# ============================================================

type CreditMap:
  CMapCons { time: u24, credit: f24, ~tail: CreditMap }
  CMapNil

def credit_map_get(map: CreditMap, time: u24) -> f24:
  match map:
    case CreditMap/CMapCons:
      if map.time == time:
        return map.credit
      else:
        return credit_map_get(map.tail, time)
    case CreditMap/CMapNil:
      return 0.0

# ============================================================
# Policy Gradient Update
# ============================================================

# Update policy using credits from interaction net reduction
def policy_gradient_update(policy: Policy, traj: Trajectory, credits: CreditMap, lr: f24) -> Policy:
  match policy:
    case Policy/PolicyNet:
      # Compute policy gradient
      grad = compute_policy_gradient(traj, credits)

      # Update weights
      new_weights = tree_update_policy(policy.weights, grad, lr)

      return Policy/PolicyNet {
        weights: new_weights,
        biases: policy.biases,
        input_size: policy.input_size,
        hidden_size: policy.hidden_size,
        output_size: policy.output_size
      }

def compute_policy_gradient(traj: Trajectory, credits: CreditMap) -> Tensor:
  match traj:
    case Trajectory/TCons:
      match traj.head:
        case TrajectoryStep/Step:
          # Gradient contribution from this step
          advantage = credit_map_get(credits, traj.head.time)
          # grad = -log_prob * advantage (for gradient ascent)
          step_grad = 0.0 - traj.head.log_prob * advantage

          # Sum gradients from all steps
          rest_grad = compute_policy_gradient(traj.tail, credits)
          return tree_add_scalar(rest_grad, step_grad)
    case Trajectory/TNil:
      return Tensor/Leaf { val: 0.0 }

def tree_update_policy(weights: Tensor, grad: Tensor, lr: f24) -> Tensor:
  match weights:
    case Tensor/Node:
      match grad:
        case Tensor/Node:
          return Tensor/Node {
            left: tree_update_policy(weights.left, grad.left, lr),
            right: tree_update_policy(weights.right, grad.right, lr)
          }
        case Tensor/Leaf:
          return Tensor/Node {
            left: tree_update_policy(weights.left, grad, lr),
            right: tree_update_policy(weights.right, grad, lr)
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
# Credit Assignment (Simplified for RL Loop)
# ============================================================

# Assign credit using simplified backward pass
# Returns credit map with advantage estimates
def assign_credit_simple(traj: Trajectory, gamma: f24) -> CreditMap:
  total_reward = trajectory_total_reward(traj)
  length = trajectory_length(traj)
  return assign_credit_rec(traj, 0, length, total_reward, gamma, CreditMap/CMapNil)

def assign_credit_rec(traj: Trajectory, t: u24, length: u24, total_reward: f24, gamma: f24, acc: CreditMap) -> CreditMap:
  match traj:
    case Trajectory/TCons:
      match traj.head:
        case TrajectoryStep/Step:
          # Credit decays with distance from reward
          distance = length - t
          discount = pow_gamma(gamma, distance)
          credit = total_reward * discount

          new_acc = CreditMap/CMapCons {
            time: traj.head.time,
            credit: credit,
            tail: acc
          }
          return assign_credit_rec(traj.tail, t + 1, length, total_reward, gamma, new_acc)
    case Trajectory/TNil:
      return acc

def pow_gamma(base: f24, exp: u24) -> f24:
  if exp == 0:
    return 1.0
  else:
    if exp == 1:
      return base
    else:
      half = pow_gamma(base, exp / 2)
      if exp % 2 == 0:
        return half * half
      else:
        return half * half * base

# ============================================================
# Episode Collection
# ============================================================

def collect_episode(policy: Policy, env: Environment, seed: u24) -> (Trajectory, f24):
  return collect_episode_rec(policy, env, Trajectory/TNil, 0, seed)

def collect_episode_rec(policy: Policy, env: Environment, traj: Trajectory, t: u24, seed: u24) -> (Trajectory, f24):
  if env_is_done(env) == 1:
    return (trajectory_reverse(traj), trajectory_total_reward(traj))
  else:
    state = env_get_state(env)

    # Sample action from policy
    sample_result = policy_sample(policy, state, seed + t)
    match sample_result:
      case (action, log_prob, logits):
        # Take action in environment
        step_result = env_step(env, action)
        match step_result:
          case (new_env, reward):
            next_state = env_get_state(new_env)
            is_done = env_is_done(new_env)

            # Create trajectory step
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

            # Add to trajectory and continue
            new_traj = trajectory_add(traj, step)
            return collect_episode_rec(policy, new_env, new_traj, t + 1, seed)

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
# Full Training Loop
# ============================================================

def train_rl_agent(policy: Policy, episodes: u24, lr: f24, gamma: f24, max_steps: u24) -> TrainState:
  return train_loop(policy, 0, episodes, lr, gamma, max_steps, 0.0, 0.0)

def train_loop(policy: Policy, ep: u24, total_eps: u24, lr: f24, gamma: f24, max_steps: u24, total_reward: f24, running_avg: f24) -> TrainState:
  if ep >= total_eps:
    return TrainState/State {
      policy: policy,
      episode: ep,
      total_reward: total_reward,
      avg_reward: running_avg
    }
  else:
    # Reset environment
    env = env_reset(max_steps, ep * 1000)

    # Collect episode
    episode_result = collect_episode(policy, env, ep * 1000 + 500)
    match episode_result:
      case (traj, ep_reward):
        # Credit assignment (this is where the magic happens!)
        credits = assign_credit_simple(traj, gamma)

        # Policy gradient update
        updated_policy = policy_gradient_update(policy, traj, credits, lr)

        # Update running average
        alpha = 0.1
        new_avg = alpha * ep_reward + (1.0 - alpha) * running_avg

        # Continue training
        return train_loop(updated_policy, ep + 1, total_eps, lr, gamma, max_steps, total_reward + ep_reward, new_avg)

# ============================================================
# Convenience Functions
# ============================================================

# Initialize and run training
def run_training(state_size: u24, hidden_size: u24, num_actions: u24, episodes: u24, lr: f24, gamma: f24, max_steps: u24, seed: u24) -> TrainState:
  policy = policy_init(state_size, hidden_size, num_actions, seed)
  return train_rl_agent(policy, episodes, lr, gamma, max_steps)

# Get final average reward
def get_avg_reward(state: TrainState) -> f24:
  match state:
    case TrainState/State:
      return state.avg_reward

# Get trained policy
def get_policy(state: TrainState) -> Policy:
  match state:
    case TrainState/State:
      return state.policy
