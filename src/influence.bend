# Influence Model for Credit Assignment
# Learns the causal influence weights for the interaction net
#
# The influence model answers: "How much did action matter?"
# It's trained with self-supervision on trajectory data:
#   influence(s, a, s') = sigmoid(error_without_a - error_with_a)
#
# If knowing the action helps predict the next state,
# then the action had causal influence on the transition.

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
# Influence Model Architecture
# ============================================================

# Simple MLP for influence prediction
# Takes (state, action, next_state) and predicts influence in [0, 1]
type InfluenceModel:
  Model {
    state_encoder: Tensor,      # Weights for encoding state
    action_encoder: Tensor,     # Weights for encoding action
    joint_encoder: Tensor,      # Weights for joint (s, a, s') encoding
    predictor: Tensor,          # Final influence predictor
    encoder_size: u24,          # Hidden dimension
    state_size: u24,            # State embedding size
    num_actions: u24            # Number of discrete actions
  }

# ============================================================
# Model Initialization
# ============================================================

# Create influence model with random initialization
def influence_model_init(state_size: u24, num_actions: u24, hidden_size: u24, seed: u24) -> InfluenceModel:
  return InfluenceModel/Model {
    state_encoder: init_weights_influence(state_size, hidden_size, seed),
    action_encoder: init_weights_influence(num_actions, hidden_size, seed + 1000),
    joint_encoder: init_weights_influence(hidden_size * 3, hidden_size, seed + 2000),
    predictor: init_weights_influence(hidden_size, 1, seed + 3000),
    encoder_size: hidden_size,
    state_size: state_size,
    num_actions: num_actions
  }

# Xavier-like weight initialization
def init_weights_influence(input_size: u24, output_size: u24, seed: u24) -> Tensor:
  total = input_size * output_size
  scale = 1.0 / sqrt_approx(input_size)
  bend i=0, s=seed:
    when i < total:
      w = lcg_random(s)
      result = Tensor/Node {
        left: Tensor/Leaf { val: (w - 0.5) * scale },
        right: fork(i + 1, s + 1)
      }
    else:
      result = Tensor/Nil
  return result

def lcg_random(seed: u24) -> f24:
  a = 1103515245
  c = 12345
  m = 2147483648
  next_seed = (a * seed + c) % m
  return next_seed / m

def sqrt_approx(x: u24) -> f24:
  # Newton's method for sqrt
  if x == 0:
    return 1.0
  else:
    guess = x / 2
    result = (guess + x / guess) / 2
    return result

# ============================================================
# Forward Pass: Compute Influence
# ============================================================

# Main influence computation
# Returns influence weight in [0, 1]
def influence_forward(model: InfluenceModel, state: Tensor, action: u24, next_state: Tensor) -> f24:
  match model:
    case InfluenceModel/Model:
      # Encode state
      state_emb = encode_tensor(model.state_encoder, state, model.encoder_size)

      # Encode action (one-hot then project)
      action_onehot = one_hot_encode(action, model.num_actions)
      action_emb = encode_tensor(model.action_encoder, action_onehot, model.encoder_size)

      # Encode next state
      next_state_emb = encode_tensor(model.state_encoder, next_state, model.encoder_size)

      # Concatenate embeddings
      joint = tensor_concat(state_emb, tensor_concat(action_emb, next_state_emb))

      # Joint encoding
      joint_emb = encode_tensor_relu(model.joint_encoder, joint, model.encoder_size)

      # Final prediction (sigmoid output)
      logit = tensor_dot_single(model.predictor, joint_emb)
      return sigmoid_influence(logit)

# Simple tensor encoding (linear layer)
def encode_tensor(weights: Tensor, input: Tensor, output_size: u24) -> Tensor:
  return tree_weighted_sum_influence(weights, input)

# Encoding with ReLU activation
def encode_tensor_relu(weights: Tensor, input: Tensor, output_size: u24) -> Tensor:
  linear = tree_weighted_sum_influence(weights, input)
  return tree_relu_influence(linear)

def tree_weighted_sum_influence(weights: Tensor, input: Tensor) -> Tensor:
  match weights:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_weighted_sum_influence(weights.left, input),
        right: tree_weighted_sum_influence(weights.right, input)
      }
    case Tensor/Leaf:
      match input:
        case Tensor/Leaf:
          return Tensor/Leaf { val: weights.val * input.val }
        case _:
          return Tensor/Leaf { val: 0.0 }
    case Tensor/Nil:
      return Tensor/Nil

def tree_relu_influence(t: Tensor) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_relu_influence(t.left),
        right: tree_relu_influence(t.right)
      }
    case Tensor/Leaf:
      if t.val > 0.0:
        return Tensor/Leaf { val: t.val }
      else:
        return Tensor/Leaf { val: 0.0 }
    case Tensor/Nil:
      return Tensor/Nil

# One-hot encoding for discrete action
def one_hot_encode(action: u24, num_actions: u24) -> Tensor:
  bend i=0:
    when i < num_actions:
      val = if i == action: 1.0 else: 0.0
      result = Tensor/Node {
        left: Tensor/Leaf { val: val },
        right: fork(i + 1)
      }
    else:
      result = Tensor/Nil
  return result

# Concatenate two tensors
def tensor_concat(a: Tensor, b: Tensor) -> Tensor:
  match a:
    case Tensor/Node:
      return Tensor/Node {
        left: a.left,
        right: tensor_concat(a.right, b)
      }
    case Tensor/Leaf:
      return Tensor/Node {
        left: a,
        right: b
      }
    case Tensor/Nil:
      return b

# Dot product for final output (single value)
def tensor_dot_single(weights: Tensor, input: Tensor) -> f24:
  fold weights with input:
    case Tensor/Node:
      match input:
        case Tensor/Node:
          return weights.left(input.left) + weights.right(input.right)
        case _:
          return 0.0
    case Tensor/Leaf:
      match input:
        case Tensor/Leaf:
          return weights.val * input.val
        case _:
          return 0.0
    case Tensor/Nil:
      return 0.0

def sigmoid_influence(x: f24) -> f24:
  if x > 10.0:
    return 0.99
  else:
    if x < -10.0:
      return 0.01
    else:
      # Sigmoid approximation using Taylor series
      ex = exp_approx_influence(x)
      return ex / (1.0 + ex)

def exp_approx_influence(x: f24) -> f24:
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

# ============================================================
# Counterfactual Influence Prediction
# ============================================================

# Predict next state given (state, action)
def predict_next_state(model: InfluenceModel, state: Tensor, action: u24) -> Tensor:
  match model:
    case InfluenceModel/Model:
      state_emb = encode_tensor(model.state_encoder, state, model.encoder_size)
      action_onehot = one_hot_encode(action, model.num_actions)
      action_emb = encode_tensor(model.action_encoder, action_onehot, model.encoder_size)

      # Predict using joint encoder
      joint = tensor_concat(state_emb, action_emb)
      return encode_tensor_relu(model.joint_encoder, joint, model.encoder_size)

# Predict next state WITHOUT action (baseline)
def predict_next_state_baseline(model: InfluenceModel, state: Tensor) -> Tensor:
  match model:
    case InfluenceModel/Model:
      state_emb = encode_tensor(model.state_encoder, state, model.encoder_size)
      # Use zero action embedding
      zero_action = tensor_zeros(model.encoder_size)
      joint = tensor_concat(state_emb, zero_action)
      return encode_tensor_relu(model.joint_encoder, joint, model.encoder_size)

def tensor_zeros(size: u24) -> Tensor:
  bend i=0:
    when i < size:
      result = Tensor/Node {
        left: Tensor/Leaf { val: 0.0 },
        right: fork(i + 1)
      }
    else:
      result = Tensor/Nil
  return result

# Compute influence as prediction improvement
# influence(s, a, s') = sigmoid(error_without - error_with)
def compute_counterfactual_influence(model: InfluenceModel, state: Tensor, action: u24, next_state: Tensor) -> f24:
  # Prediction with action knowledge
  pred_with = predict_next_state(model, state, action)
  error_with = prediction_error(pred_with, next_state)

  # Prediction without action (baseline)
  pred_without = predict_next_state_baseline(model, state)
  error_without = prediction_error(pred_without, next_state)

  # Influence = how much action improved prediction
  improvement = error_without - error_with
  return sigmoid_influence(improvement)

def prediction_error(pred: Tensor, target: Tensor) -> f24:
  squared_diff = tree_squared_diff(pred, target)
  return tree_sum_influence(squared_diff)

def tree_squared_diff(a: Tensor, b: Tensor) -> Tensor:
  match a:
    case Tensor/Node:
      match b:
        case Tensor/Node:
          return Tensor/Node {
            left: tree_squared_diff(a.left, b.left),
            right: tree_squared_diff(a.right, b.right)
          }
        case _:
          return Tensor/Nil
    case Tensor/Leaf:
      match b:
        case Tensor/Leaf:
          diff = a.val - b.val
          return Tensor/Leaf { val: diff * diff }
        case _:
          return Tensor/Nil
    case Tensor/Nil:
      return Tensor/Nil

def tree_sum_influence(t: Tensor) -> f24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return t.val
    case Tensor/Nil:
      return 0.0

# ============================================================
# Training: Self-Supervised on Trajectory Data
# ============================================================

# Training step type
type InfluenceTrainState:
  IState {
    model: InfluenceModel,
    learning_rate: f24,
    total_loss: f24,
    steps: u24
  }

# Trajectory step for training
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

# Train influence model on a trajectory
def influence_train(state: InfluenceTrainState, traj: Trajectory) -> InfluenceTrainState:
  match traj:
    case Trajectory/TCons:
      # Train on this step
      new_state = influence_train_step(state, traj.head)
      # Continue with rest
      return influence_train(new_state, traj.tail)
    case Trajectory/TNil:
      return state

# Single training step
def influence_train_step(state: InfluenceTrainState, step: TrajectoryStep) -> InfluenceTrainState:
  match state:
    case InfluenceTrainState/IState:
      match step:
        case TrajectoryStep/Step:
          # Forward pass
          influence = influence_forward(state.model, step.state, step.action, step.next_state)

          # Target: high influence if state changed significantly
          target = compute_target_influence(step.state, step.next_state)

          # Compute loss (MSE)
          diff = influence - target
          loss = diff * diff

          # Backward pass and update (simplified)
          updated_model = influence_backward_update(state.model, step, diff, state.learning_rate)

          return InfluenceTrainState/IState {
            model: updated_model,
            learning_rate: state.learning_rate,
            total_loss: state.total_loss + loss,
            steps: state.steps + 1
          }

# Compute target influence from state change
def compute_target_influence(state: Tensor, next_state: Tensor) -> f24:
  change = tree_sum_influence(tree_squared_diff(state, next_state))
  # Normalize to [0, 1]
  return sigmoid_influence(change - 1.0)

# Simplified backward pass
def influence_backward_update(model: InfluenceModel, step: TrajectoryStep, error: f24, lr: f24) -> InfluenceModel:
  match model:
    case InfluenceModel/Model:
      # Update predictor weights based on error
      updated_predictor = tree_update_weights(model.predictor, error, lr)

      # Update joint encoder
      updated_joint = tree_update_weights(model.joint_encoder, error * 0.5, lr)

      return InfluenceModel/Model {
        state_encoder: model.state_encoder,
        action_encoder: model.action_encoder,
        joint_encoder: updated_joint,
        predictor: updated_predictor,
        encoder_size: model.encoder_size,
        state_size: model.state_size,
        num_actions: model.num_actions
      }

def tree_update_weights(weights: Tensor, grad: f24, lr: f24) -> Tensor:
  match weights:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_update_weights(weights.left, grad, lr),
        right: tree_update_weights(weights.right, grad, lr)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: weights.val - lr * grad }
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# Batch Influence Computation for Trajectory
# ============================================================

# Compute influences for entire trajectory (parallel)
type InfluenceList:
  ICons { time: u24, influence: f24, ~tail: InfluenceList }
  INil

def compute_trajectory_influences(model: InfluenceModel, traj: Trajectory) -> InfluenceList:
  match traj:
    case Trajectory/TCons:
      match traj.head:
        case TrajectoryStep/Step:
          inf = influence_forward(model, traj.head.state, traj.head.action, traj.head.next_state)
          return InfluenceList/ICons {
            time: traj.head.time,
            influence: inf,
            tail: compute_trajectory_influences(model, traj.tail)
          }
    case Trajectory/TNil:
      return InfluenceList/INil

# Lookup influence for timestep
def influence_list_get(list: InfluenceList, time: u24) -> f24:
  match list:
    case InfluenceList/ICons:
      if list.time == time:
        return list.influence
      else:
        return influence_list_get(list.tail, time)
    case InfluenceList/INil:
      return 1.0  # Default to full influence

# ============================================================
# Model Utilities
# ============================================================

# Get average loss
def influence_avg_loss(state: InfluenceTrainState) -> f24:
  match state:
    case InfluenceTrainState/IState:
      if state.steps > 0:
        return state.total_loss / state.steps
      else:
        return 0.0

# Count model parameters
def influence_param_count(model: InfluenceModel) -> u24:
  match model:
    case InfluenceModel/Model:
      state_params = tree_count(model.state_encoder)
      action_params = tree_count(model.action_encoder)
      joint_params = tree_count(model.joint_encoder)
      pred_params = tree_count(model.predictor)
      return state_params + action_params + joint_params + pred_params

def tree_count(t: Tensor) -> u24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return 1
    case Tensor/Nil:
      return 0
