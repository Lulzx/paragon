# MNIST with Fully Parallel Tree Operations
# Key insight: Align ALL structures as trees for parallel operations
# - Weights are a tree of trees (matching output x input structure)
# - Gradients computed via parallel fold
# - Updates via parallel tree zip

type Tensor:
  Node { ~left: Tensor, ~right: Tensor }
  Leaf { val: f24 }
  Nil

# ============================================================
# Core Parallel Operations - All O(log n)
# ============================================================

def tree_sum(t: Tensor) -> f24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return t.val
    case Tensor/Nil:
      return 0.0

def tree_dot(a: Tensor, b: Tensor) -> f24:
  fold a with b:
    case Tensor/Node:
      match b:
        case Tensor/Node:
          return a.left(b.left) + a.right(b.right)
        case _:
          return 0.0
    case Tensor/Leaf:
      match b:
        case Tensor/Leaf:
          return a.val * b.val
        case _:
          return 0.0
    case Tensor/Nil:
      return 0.0

def tree_add(a: Tensor, b: Tensor) -> Tensor:
  match a:
    case Tensor/Node:
      match b:
        case Tensor/Node:
          return Tensor/Node {
            left: tree_add(a.left, b.left),
            right: tree_add(a.right, b.right)
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

def tree_sub(a: Tensor, b: Tensor) -> Tensor:
  match a:
    case Tensor/Node:
      match b:
        case Tensor/Node:
          return Tensor/Node {
            left: tree_sub(a.left, b.left),
            right: tree_sub(a.right, b.right)
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

def tree_scale(t: Tensor, s: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_scale(t.left, s),
        right: tree_scale(t.right, s)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: t.val * s }
    case Tensor/Nil:
      return Tensor/Nil

def tree_size(t: Tensor) -> u24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return 1
    case Tensor/Nil:
      return 0

def lcg(seed: u24) -> u24:
  return (seed * 1103 + 12345) % 100000

# ============================================================
# Softmax - Parallel
# ============================================================

def softmax(t: Tensor) -> Tensor:
  max_val = tree_max(t)
  exp_t = tree_exp_shift(t, max_val)
  total = tree_sum(exp_t)
  if total > 0.0001:
    return tree_scale(exp_t, 1.0 / total)
  else:
    return exp_t

def tree_max(t: Tensor) -> f24:
  fold t:
    case Tensor/Node:
      if t.left > t.right:
        return t.left
      else:
        return t.right
    case Tensor/Leaf:
      return t.val
    case Tensor/Nil:
      return 0.0 - 999999.0

def tree_exp_shift(t: Tensor, shift: f24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_exp_shift(t.left, shift),
        right: tree_exp_shift(t.right, shift)
      }
    case Tensor/Leaf:
      x = t.val - shift
      return Tensor/Leaf { val: exp_approx(x) }
    case Tensor/Nil:
      return Tensor/Nil

def exp_approx(x: f24) -> f24:
  if x > 10.0:
    return 22026.0
  else:
    if x < 0.0 - 10.0:
      return 0.00005
    else:
      x2 = x * x
      x3 = x2 * x
      x4 = x3 * x
      return 1.0 + x + x2 * 0.5 + x3 * 0.1667 + x4 * 0.0417

# ============================================================
# Weight Structure: Tree of Weight Vectors
# WeightTree matches output structure (10 classes -> tree of 10)
# Each node contains a weight vector (tree matching input structure)
# ============================================================

type WeightTree:
  WNode { ~left: WeightTree, ~right: WeightTree }
  WLeaf { weights: Tensor }  # A full weight vector for one class
  WNil

type Model:
  Net { weight_tree: WeightTree, biases: Tensor }

# Create model: weight_tree has same structure as output (10 classes)
def create_model(input_size: u24, output_size: u24, seed: u24) -> Model:
  wt = create_weight_tree(output_size, input_size, seed)
  biases = create_zeros(output_size)
  return Model/Net { weight_tree: wt, biases: biases }

def create_weight_tree(output_size: u24, input_size: u24, seed: u24) -> WeightTree:
  if output_size == 0:
    return WeightTree/WNil
  else:
    if output_size == 1:
      return WeightTree/WLeaf { weights: create_weights(input_size, seed) }
    else:
      half = output_size / 2
      return WeightTree/WNode {
        left: create_weight_tree(half, input_size, seed),
        right: create_weight_tree(output_size - half, input_size, lcg(seed + 1000))
      }

def create_weights(size: u24, seed: u24) -> Tensor:
  if size == 0:
    return Tensor/Nil
  else:
    if size == 1:
      w = u24/to_f24(lcg(seed) % 100) / 500.0 - 0.1
      return Tensor/Leaf { val: w }
    else:
      half = size / 2
      return Tensor/Node {
        left: create_weights(half, seed),
        right: create_weights(size - half, lcg(seed + half))
      }

def create_zeros(size: u24) -> Tensor:
  if size == 0:
    return Tensor/Nil
  else:
    if size == 1:
      return Tensor/Leaf { val: 0.0 }
    else:
      half = size / 2
      return Tensor/Node {
        left: create_zeros(half),
        right: create_zeros(size - half)
      }

# ============================================================
# Forward: Parallel over all classes via fold
# ============================================================

def forward(model: Model, input: Tensor) -> Tensor:
  match model:
    case Model/Net:
      scores = compute_all_scores(model.weight_tree, input)
      scores_biased = tree_add(scores, model.biases)
      return softmax(scores_biased)

# Compute scores for ALL classes in parallel via fold
def compute_all_scores(wt: WeightTree, input: Tensor) -> Tensor:
  fold wt with input:
    case WeightTree/WNode:
      return Tensor/Node {
        left: wt.left(input),
        right: wt.right(input)
      }
    case WeightTree/WLeaf:
      # Dot product of this class's weights with input
      score = tree_dot(wt.weights, input)
      return Tensor/Leaf { val: score }
    case WeightTree/WNil:
      return Tensor/Nil

# ============================================================
# Backward: Parallel gradient update via fold
# Key: d_output has same tree structure as weight_tree
# So we can zip them together for parallel updates!
# ============================================================

def train_step(model: Model, input: Tensor, target: Tensor, lr: f24) -> Model:
  match model:
    case Model/Net:
      output = forward(model, input)
      d_output = tree_sub(output, target)

      # Update weights: parallel zip of weight_tree and d_output
      new_wt = update_weight_tree(model.weight_tree, d_output, input, lr)
      new_biases = tree_sub(model.biases, tree_scale(d_output, lr))

      return Model/Net { weight_tree: new_wt, biases: new_biases }

# Parallel update: fold over weight_tree with d_output (same structure!)
def update_weight_tree(wt: WeightTree, d_out: Tensor, input: Tensor, lr: f24) -> WeightTree:
  match wt:
    case WeightTree/WNode:
      match d_out:
        case Tensor/Node:
          return WeightTree/WNode {
            left: update_weight_tree(wt.left, d_out.left, input, lr),
            right: update_weight_tree(wt.right, d_out.right, input, lr)
          }
        case _:
          return wt
    case WeightTree/WLeaf:
      match d_out:
        case Tensor/Leaf:
          # grad = d * input (outer product becomes scalar * vector)
          grad = tree_scale(input, d_out.val)
          new_weights = tree_sub(wt.weights, tree_scale(grad, lr))
          return WeightTree/WLeaf { weights: new_weights }
        case _:
          return wt
    case WeightTree/WNil:
      return WeightTree/WNil

# ============================================================
# Training Loop
# ============================================================

def train(model: Model, epochs: u24, lr: f24, seed: u24) -> Model:
  return train_loop(model, epochs, 0, lr, seed)

def train_loop(model: Model, total: u24, current: u24, lr: f24, seed: u24) -> Model:
  if current >= total:
    return model
  else:
    digit = current % 10
    input = generate_digit_pattern(digit, seed + current * 17)
    target = one_hot(digit, 10)
    new_model = train_step(model, input, target, lr)
    return train_loop(new_model, total, current + 1, lr, seed)

# ============================================================
# Data: Strong digit-specific patterns in tree structure
# Each digit activates specific tree regions
# ============================================================

def generate_digit_pattern(digit: u24, seed: u24) -> Tensor:
  return build_pattern_tree(digit, seed, 16, 0)

def build_pattern_tree(digit: u24, seed: u24, size: u24, position: u24) -> Tensor:
  if size == 0:
    return Tensor/Nil
  else:
    if size == 1:
      noise = u24/to_f24(lcg(seed) % 20) / 200.0
      val = compute_feature(digit, position, noise)
      return Tensor/Leaf { val: val }
    else:
      half = size / 2
      return Tensor/Node {
        left: build_pattern_tree(digit, seed, half, position),
        right: build_pattern_tree(digit, lcg(seed + 1), size - half, position + half)
      }

def compute_feature(digit: u24, position: u24, noise: f24) -> f24:
  # Strong pattern: digit d is strong at positions that hash to d
  # Use multiple positions per digit for redundancy
  pos_hash = (position * 7 + 3) % 10
  if pos_hash == digit:
    return 0.95 + noise
  else:
    pos_hash2 = (position * 3 + 7) % 10
    if pos_hash2 == digit:
      return 0.85 + noise
    else:
      return 0.05 + noise

def one_hot(target: u24, size: u24) -> Tensor:
  return build_one_hot(target, size, 0)

def build_one_hot(target: u24, size: u24, idx: u24) -> Tensor:
  if size == 0:
    return Tensor/Nil
  else:
    if size == 1:
      if idx == target:
        return Tensor/Leaf { val: 1.0 }
      else:
        return Tensor/Leaf { val: 0.0 }
    else:
      half = size / 2
      return Tensor/Node {
        left: build_one_hot(target, half, idx),
        right: build_one_hot(target, size - half, idx + half)
      }

# ============================================================
# Prediction & Evaluation
# ============================================================

def predict(model: Model, input: Tensor) -> u24:
  output = forward(model, input)
  return argmax(output)

def argmax(t: Tensor) -> u24:
  (max_val, max_idx) = find_max(t, 0)
  return max_idx

def find_max(t: Tensor, offset: u24) -> (f24, u24):
  match t:
    case Tensor/Node:
      (left_max, left_idx) = find_max(t.left, offset)
      left_size = tree_size(t.left)
      (right_max, right_idx) = find_max(t.right, offset + left_size)
      if left_max > right_max:
        return (left_max, left_idx)
      else:
        return (right_max, right_idx)
    case Tensor/Leaf:
      return (t.val, offset)
    case Tensor/Nil:
      return (0.0 - 999999.0, offset)

def evaluate(model: Model, num_test: u24, seed: u24) -> u24:
  return eval_loop(model, num_test, 0, seed, 0)

def eval_loop(model: Model, total: u24, current: u24, seed: u24, correct: u24) -> u24:
  if current >= total:
    return correct
  else:
    digit = current % 10
    input = generate_digit_pattern(digit, seed + current * 31 + 99999)
    pred = predict(model, input)
    new_correct = correct
    if pred == digit:
      new_correct = correct + 1
    else:
      new_correct = correct
    return eval_loop(model, total, current + 1, seed, new_correct)

# ============================================================
# Main
# ============================================================

def main() -> f24:
  # 16 input features, 10 output classes
  model = create_model(16, 10, 42)

  # Train: 5000 samples (500 per digit) for robust learning
  trained = train(model, 5000, 0.2, 7777)

  # Test: 500 samples with different seed
  correct = evaluate(trained, 500, 12121)

  # Return accuracy percentage
  return u24/to_f24(correct) / 5.0
