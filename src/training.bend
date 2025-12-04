# Training utilities for neural networks
# Includes batch processing, data handling, metrics, and learning rate scheduling

type Tensor:
  Node { ~left: Tensor, ~right: Tensor }
  Leaf { val: f24 }
  Nil

# ============================================================
# Dataset Types
# ============================================================

# Single data sample
type Sample:
  Data { input: Tensor, target: Tensor }
  Empty

# Dataset as a list of samples
type Dataset:
  List { head: Sample, ~tail: Dataset }
  Nil

# Batch of samples
type Batch:
  BatchData {
    inputs: BatchTensor,    # List of input tensors
    targets: BatchTensor,   # List of target tensors
    size: u24               # Number of samples in batch
  }
  Empty

type BatchTensor:
  BTensor { head: Tensor, ~tail: BatchTensor }
  Nil

# ============================================================
# Dataset Operations
# ============================================================

# Get dataset length
def dataset_length(ds: Dataset) -> u24:
  fold ds:
    case Dataset/List:
      return 1 + ds.tail
    case Dataset/Nil:
      return 0

# Append sample to dataset
def dataset_append(ds: Dataset, sample: Sample) -> Dataset:
  match ds:
    case Dataset/List:
      return Dataset/List {
        head: ds.head,
        tail: dataset_append(ds.tail, sample)
      }
    case Dataset/Nil:
      return Dataset/List { head: sample, tail: Dataset/Nil }

# Get sample at index
def dataset_get(ds: Dataset, idx: u24) -> Sample:
  match ds:
    case Dataset/List:
      if idx == 0:
        return ds.head
      else:
        return dataset_get(ds.tail, idx - 1)
    case Dataset/Nil:
      return Sample/Empty

# ============================================================
# Batching
# ============================================================

# Create batches from dataset
def create_batches(ds: Dataset, batch_size: u24) -> BatchList:
  return create_batches_helper(ds, batch_size, 0, dataset_length(ds))

type BatchList:
  BList { head: Batch, ~tail: BatchList }
  Nil

def create_batches_helper(ds: Dataset, batch_size: u24, start: u24, total: u24) -> BatchList:
  if start >= total:
    return BatchList/Nil
  else:
    end = start + batch_size
    if end > total:
      end = total
    batch = extract_batch(ds, start, end)
    return BatchList/BList {
      head: batch,
      tail: create_batches_helper(ds, batch_size, end, total)
    }

# Extract batch from dataset [start, end)
def extract_batch(ds: Dataset, start: u24, end: u24) -> Batch:
  inputs = extract_inputs(ds, start, end)
  targets = extract_targets(ds, start, end)
  return Batch/BatchData {
    inputs: inputs,
    targets: targets,
    size: end - start
  }

def extract_inputs(ds: Dataset, start: u24, end: u24) -> BatchTensor:
  bend i=start:
    when i < end:
      sample = dataset_get(ds, i)
      match sample:
        case Sample/Data:
          result = BatchTensor/BTensor {
            head: sample.input,
            tail: fork(i + 1)
          }
        case Sample/Empty:
          result = BatchTensor/Nil
    else:
      result = BatchTensor/Nil
  return result

def extract_targets(ds: Dataset, start: u24, end: u24) -> BatchTensor:
  bend i=start:
    when i < end:
      sample = dataset_get(ds, i)
      match sample:
        case Sample/Data:
          result = BatchTensor/BTensor {
            head: sample.target,
            tail: fork(i + 1)
          }
        case Sample/Empty:
          result = BatchTensor/Nil
    else:
      result = BatchTensor/Nil
  return result

# ============================================================
# Batch Gradient Accumulation
# ============================================================

# Accumulate gradients over a batch
def accumulate_gradients(grad_list: BatchTensor) -> Tensor:
  # Sum all gradients and divide by count
  sum_grad = batch_sum_tensors(grad_list)
  count = batch_tensor_length(grad_list)
  if count > 0:
    return tree_scale_train(sum_grad, 1.0 / count)
  else:
    return sum_grad

def batch_sum_tensors(bt: BatchTensor) -> Tensor:
  match bt:
    case BatchTensor/BTensor:
      rest = batch_sum_tensors(bt.tail)
      return tree_add_train(bt.head, rest)
    case BatchTensor/Nil:
      return Tensor/Nil

def tree_add_train(a: Tensor, b: Tensor) -> Tensor:
  match a:
    case Tensor/Node:
      match b:
        case Tensor/Node:
          return Tensor/Node {
            left: tree_add_train(a.left, b.left),
            right: tree_add_train(a.right, b.right)
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

def tree_scale_train(t: Tensor, s: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_scale_train(t.left, s),
        right: tree_scale_train(t.right, s)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: t.val * s }
    case Tensor/Nil:
      return Tensor/Nil

def batch_tensor_length(bt: BatchTensor) -> u24:
  fold bt:
    case BatchTensor/BTensor:
      return 1 + bt.tail
    case BatchTensor/Nil:
      return 0

# ============================================================
# Learning Rate Scheduling
# ============================================================

# Scheduler types
type LRScheduler:
  Constant { lr: f24 }
  StepDecay { initial_lr: f24, decay_rate: f24, step_size: u24 }
  ExponentialDecay { initial_lr: f24, decay_rate: f24 }
  CosineAnnealing { initial_lr: f24, min_lr: f24, t_max: u24 }
  WarmupCosine { initial_lr: f24, warmup_steps: u24, total_steps: u24 }
  ReduceOnPlateau { lr: f24, factor: f24, patience: u24, min_lr: f24, best_loss: f24, counter: u24 }
  Linear { initial_lr: f24, final_lr: f24, total_steps: u24 }

# Get learning rate for current step/epoch
def get_lr(scheduler: LRScheduler, step: u24) -> f24:
  match scheduler:
    case LRScheduler/Constant:
      return scheduler.lr
    case LRScheduler/StepDecay:
      num_decays = step / scheduler.step_size
      return scheduler.initial_lr * pow_decay(scheduler.decay_rate, num_decays)
    case LRScheduler/ExponentialDecay:
      return scheduler.initial_lr * pow_decay(scheduler.decay_rate, step)
    case LRScheduler/CosineAnnealing:
      return cosine_lr(scheduler.initial_lr, scheduler.min_lr, step, scheduler.t_max)
    case LRScheduler/WarmupCosine:
      return warmup_cosine_lr(scheduler.initial_lr, scheduler.warmup_steps, scheduler.total_steps, step)
    case LRScheduler/ReduceOnPlateau:
      return scheduler.lr
    case LRScheduler/Linear:
      return linear_lr(scheduler.initial_lr, scheduler.final_lr, step, scheduler.total_steps)

# Step decay: lr * decay^(step / step_size)
def pow_decay(base: f24, exp: u24) -> f24:
  bend i=0, result=1.0:
    when i < exp:
      fork(i + 1, result * base)
    else:
      result = result
  return result

# Cosine annealing: min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * t / t_max))
def cosine_lr(max_lr: f24, min_lr: f24, step: u24, t_max: u24) -> f24:
  if t_max == 0:
    return max_lr
  else:
    progress = step / t_max
    # Cosine approximation: cos(pi*x) ≈ 1 - 2*x^2 for small x
    # More accurate: piecewise
    cos_val = cos_approx(progress * 3.14159)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + cos_val)

# Cosine approximation using Taylor series
def cos_approx(x: f24) -> f24:
  # Normalize x to [0, 2*pi]
  x2 = x * x
  x4 = x2 * x2
  x6 = x4 * x2
  # cos(x) ≈ 1 - x²/2 + x⁴/24 - x⁶/720
  return 1.0 - x2 * 0.5 + x4 * 0.041667 - x6 * 0.001389

# Warmup + cosine decay
def warmup_cosine_lr(max_lr: f24, warmup_steps: u24, total_steps: u24, step: u24) -> f24:
  if step < warmup_steps:
    # Linear warmup
    return max_lr * (step + 1) / warmup_steps
  else:
    # Cosine decay after warmup
    decay_steps = total_steps - warmup_steps
    decay_step = step - warmup_steps
    return cosine_lr(max_lr, 0.0, decay_step, decay_steps)

# Linear decay
def linear_lr(initial: f24, final: f24, step: u24, total: u24) -> f24:
  if step >= total:
    return final
  else:
    progress = step / total
    return initial + (final - initial) * progress

# Update ReduceOnPlateau scheduler based on loss
def update_plateau_scheduler(scheduler: LRScheduler, loss: f24) -> LRScheduler:
  match scheduler:
    case LRScheduler/ReduceOnPlateau:
      if loss < scheduler.best_loss:
        # Improvement - reset counter
        return LRScheduler/ReduceOnPlateau {
          lr: scheduler.lr,
          factor: scheduler.factor,
          patience: scheduler.patience,
          min_lr: scheduler.min_lr,
          best_loss: loss,
          counter: 0
        }
      else:
        # No improvement - increment counter
        new_counter = scheduler.counter + 1
        if new_counter >= scheduler.patience:
          # Reduce learning rate
          new_lr = scheduler.lr * scheduler.factor
          if new_lr < scheduler.min_lr:
            new_lr = scheduler.min_lr
          return LRScheduler/ReduceOnPlateau {
            lr: new_lr,
            factor: scheduler.factor,
            patience: scheduler.patience,
            min_lr: scheduler.min_lr,
            best_loss: scheduler.best_loss,
            counter: 0
          }
        else:
          return LRScheduler/ReduceOnPlateau {
            lr: scheduler.lr,
            factor: scheduler.factor,
            patience: scheduler.patience,
            min_lr: scheduler.min_lr,
            best_loss: scheduler.best_loss,
            counter: new_counter
          }
    case _:
      return scheduler

# ============================================================
# Metrics
# ============================================================

# Compute accuracy for classification
def compute_accuracy(predictions: BatchTensor, targets: BatchTensor) -> f24:
  correct = count_correct(predictions, targets)
  total = batch_tensor_length(predictions)
  if total > 0:
    return correct / total
  else:
    return 0.0

def count_correct(preds: BatchTensor, targets: BatchTensor) -> f24:
  match preds:
    case BatchTensor/BTensor:
      match targets:
        case BatchTensor/BTensor:
          is_correct = check_prediction(preds.head, targets.head)
          rest = count_correct(preds.tail, targets.tail)
          return is_correct + rest
        case BatchTensor/Nil:
          return 0.0
    case BatchTensor/Nil:
      return 0.0

# Check if prediction matches target (argmax comparison)
def check_prediction(pred: Tensor, target: Tensor) -> f24:
  pred_class = argmax_tensor(pred)
  target_class = argmax_tensor(target)
  if pred_class == target_class:
    return 1.0
  else:
    return 0.0

# Find index of maximum value
def argmax_tensor(t: Tensor) -> u24:
  result = argmax_helper(t, 0, -999999.0, 0)
  return result

def argmax_helper(t: Tensor, idx: u24, max_val: f24, max_idx: u24) -> u24:
  match t:
    case Tensor/Node:
      left_result = argmax_helper(t.left, idx, max_val, max_idx)
      left_count = tree_count_train(t.left)
      # Get the max value from left subtree for comparison
      return argmax_helper(t.right, idx + left_count, max_val, left_result)
    case Tensor/Leaf:
      if t.val > max_val:
        return idx
      else:
        return max_idx
    case Tensor/Nil:
      return max_idx

def tree_count_train(t: Tensor) -> u24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return 1
    case Tensor/Nil:
      return 0

# Compute average loss over batch
def average_batch_loss(losses: List_f24) -> f24:
  total = sum_list(losses)
  count = list_length(losses)
  if count > 0:
    return total / count
  else:
    return 0.0

type List_f24:
  Cons { head: f24, ~tail: List_f24 }
  Nil

def sum_list(l: List_f24) -> f24:
  fold l:
    case List_f24/Cons:
      return l.head + l.tail
    case List_f24/Nil:
      return 0.0

def list_length(l: List_f24) -> u24:
  fold l:
    case List_f24/Cons:
      return 1 + l.tail
    case List_f24/Nil:
      return 0

# ============================================================
# Data Preprocessing
# ============================================================

# Normalize tensor to [0, 1] range
def normalize_minmax(t: Tensor, min_val: f24, max_val: f24) -> Tensor:
  range = max_val - min_val
  if range < 0.0001:
    return t
  else:
    return tree_normalize(t, min_val, range)

def tree_normalize(t: Tensor, min_val: f24, range: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_normalize(t.left, min_val, range),
        right: tree_normalize(t.right, min_val, range)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: (t.val - min_val) / range }
    case Tensor/Nil:
      return Tensor/Nil

# Standardize tensor (mean=0, std=1)
def standardize(t: Tensor, mean: f24, std: f24) -> Tensor:
  if std < 0.0001:
    return t
  else:
    return tree_standardize(t, mean, std)

def tree_standardize(t: Tensor, mean: f24, std: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_standardize(t.left, mean, std),
        right: tree_standardize(t.right, mean, std)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: (t.val - mean) / std }
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# One-Hot Encoding
# ============================================================

# Create one-hot encoded tensor
def one_hot(class_idx: u24, num_classes: u24) -> Tensor:
  bend i=0:
    when i < num_classes:
      if i == class_idx:
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
# Data Shuffling (using Fisher-Yates with LCG)
# ============================================================

# Shuffle dataset indices
def shuffle_indices(size: u24, seed: u24) -> List_u24:
  # Create initial indices [0, 1, ..., size-1]
  indices = create_indices(size)
  # Fisher-Yates shuffle
  return fisher_yates(indices, size, seed)

type List_u24:
  Cons { head: u24, ~tail: List_u24 }
  Nil

def create_indices(size: u24) -> List_u24:
  bend i=0:
    when i < size:
      result = List_u24/Cons { head: i, tail: fork(i + 1) }
    else:
      result = List_u24/Nil
  return result

def fisher_yates(indices: List_u24, size: u24, seed: u24) -> List_u24:
  # Simplified shuffle - swap each element with random later element
  return shuffle_pass(indices, 0, size, seed)

def shuffle_pass(indices: List_u24, pos: u24, size: u24, seed: u24) -> List_u24:
  match indices:
    case List_u24/Cons:
      if pos >= size - 1:
        return indices
      else:
        # Generate random index in [pos, size)
        rand_offset = lcg_rand(seed + pos) % (size - pos)
        swap_idx = pos + rand_offset
        # Swap elements at pos and swap_idx
        swapped = swap_at(indices, pos, swap_idx, 0)
        return shuffle_pass(swapped, pos + 1, size, seed)
    case List_u24/Nil:
      return List_u24/Nil

def lcg_rand(seed: u24) -> u24:
  a = 1103515245
  c = 12345
  m = 2147483648
  return (a * seed + c) % m

def swap_at(list: List_u24, i: u24, j: u24, current: u24) -> List_u24:
  # Get values at i and j, then reconstruct with swapped values
  val_i = list_get(list, i)
  val_j = list_get(list, j)
  return list_set(list_set(list, i, val_j, 0), j, val_i, 0)

def list_get(list: List_u24, idx: u24) -> u24:
  match list:
    case List_u24/Cons:
      if idx == 0:
        return list.head
      else:
        return list_get(list.tail, idx - 1)
    case List_u24/Nil:
      return 0

def list_set(list: List_u24, idx: u24, val: u24, current: u24) -> List_u24:
  match list:
    case List_u24/Cons:
      if current == idx:
        return List_u24/Cons { head: val, tail: list.tail }
      else:
        return List_u24/Cons {
          head: list.head,
          tail: list_set(list.tail, idx, val, current + 1)
        }
    case List_u24/Nil:
      return List_u24/Nil

# ============================================================
# Early Stopping
# ============================================================

type EarlyStopState:
  State {
    best_loss: f24,
    counter: u24,
    patience: u24,
    min_delta: f24,
    stopped: u24  # 0 = continue, 1 = stop
  }
  None

def init_early_stopping(patience: u24, min_delta: f24) -> EarlyStopState:
  return EarlyStopState/State {
    best_loss: 999999.0,
    counter: 0,
    patience: patience,
    min_delta: min_delta,
    stopped: 0
  }

def check_early_stop(state: EarlyStopState, val_loss: f24) -> EarlyStopState:
  match state:
    case EarlyStopState/State:
      improvement = state.best_loss - val_loss
      if improvement > state.min_delta:
        # Significant improvement
        return EarlyStopState/State {
          best_loss: val_loss,
          counter: 0,
          patience: state.patience,
          min_delta: state.min_delta,
          stopped: 0
        }
      else:
        # No significant improvement
        new_counter = state.counter + 1
        if new_counter >= state.patience:
          return EarlyStopState/State {
            best_loss: state.best_loss,
            counter: new_counter,
            patience: state.patience,
            min_delta: state.min_delta,
            stopped: 1
          }
        else:
          return EarlyStopState/State {
            best_loss: state.best_loss,
            counter: new_counter,
            patience: state.patience,
            min_delta: state.min_delta,
            stopped: 0
          }
    case EarlyStopState/None:
      return EarlyStopState/None

def should_stop(state: EarlyStopState) -> u24:
  match state:
    case EarlyStopState/State:
      return state.stopped
    case EarlyStopState/None:
      return 0

# ============================================================
# Training History
# ============================================================

type TrainingHistory:
  History {
    train_losses: List_f24,
    val_losses: List_f24,
    train_accs: List_f24,
    val_accs: List_f24
  }

def init_history() -> TrainingHistory:
  return TrainingHistory/History {
    train_losses: List_f24/Nil,
    val_losses: List_f24/Nil,
    train_accs: List_f24/Nil,
    val_accs: List_f24/Nil
  }

def record_epoch(history: TrainingHistory, train_loss: f24, val_loss: f24, train_acc: f24, val_acc: f24) -> TrainingHistory:
  match history:
    case TrainingHistory/History:
      return TrainingHistory/History {
        train_losses: List_f24/Cons { head: train_loss, tail: history.train_losses },
        val_losses: List_f24/Cons { head: val_loss, tail: history.val_losses },
        train_accs: List_f24/Cons { head: train_acc, tail: history.train_accs },
        val_accs: List_f24/Cons { head: val_acc, tail: history.val_accs }
      }
