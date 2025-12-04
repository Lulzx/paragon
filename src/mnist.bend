# MNIST Dataset Loader and Utilities
# Provides functions for loading and preprocessing MNIST data

type Tensor:
  Node { ~left: Tensor, ~right: Tensor }
  Leaf { val: f24 }
  Nil

# ============================================================
# MNIST Constants
# ============================================================

# MNIST image dimensions
# 28x28 pixels = 784 input features
# 10 classes (digits 0-9)

def mnist_input_size() -> u24:
  return 784

def mnist_num_classes() -> u24:
  return 10

def mnist_image_width() -> u24:
  return 28

def mnist_image_height() -> u24:
  return 28

# ============================================================
# Image Representation
# ============================================================

# MNIST image as a tensor (flattened 28x28 = 784 values)
type MNISTImage:
  Image { pixels: Tensor, label: u24 }
  Empty

# MNIST dataset
type MNISTDataset:
  Dataset { images: ImageList, size: u24 }
  Empty

type ImageList:
  Cons { head: MNISTImage, ~tail: ImageList }
  Nil

# ============================================================
# Preprocessing
# ============================================================

# Normalize pixel values from [0, 255] to [0, 1]
def normalize_pixels(pixels: Tensor) -> Tensor:
  match pixels:
    case Tensor/Node:
      return Tensor/Node {
        left: normalize_pixels(pixels.left),
        right: normalize_pixels(pixels.right)
      }
    case Tensor/Leaf:
      # Divide by 255.0 to get [0, 1] range
      return Tensor/Leaf { val: pixels.val / 255.0 }
    case Tensor/Nil:
      return Tensor/Nil

# Normalize to [-0.5, 0.5] (zero-centered)
def normalize_centered(pixels: Tensor) -> Tensor:
  match pixels:
    case Tensor/Node:
      return Tensor/Node {
        left: normalize_centered(pixels.left),
        right: normalize_centered(pixels.right)
      }
    case Tensor/Leaf:
      return Tensor/Leaf { val: (pixels.val / 255.0) - 0.5 }
    case Tensor/Nil:
      return Tensor/Nil

# Standardize using MNIST dataset statistics
# Mean ≈ 0.1307, Std ≈ 0.3081
def standardize_mnist(pixels: Tensor) -> Tensor:
  mean = 0.1307
  std = 0.3081
  match pixels:
    case Tensor/Node:
      return Tensor/Node {
        left: standardize_mnist(pixels.left),
        right: standardize_mnist(pixels.right)
      }
    case Tensor/Leaf:
      normalized = pixels.val / 255.0
      return Tensor/Leaf { val: (normalized - mean) / std }
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# One-Hot Encoding for Labels
# ============================================================

# Convert label (0-9) to one-hot tensor
def label_to_onehot(label: u24) -> Tensor:
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

# Convert prediction tensor to class index (argmax)
def prediction_to_class(pred: Tensor) -> u24:
  return find_argmax(pred, 0, -999999.0, 0)

def find_argmax(t: Tensor, idx: u24, max_val: f24, max_idx: u24) -> u24:
  match t:
    case Tensor/Node:
      # Process left subtree
      left_idx = find_argmax(t.left, idx, max_val, max_idx)
      left_count = count_leaves(t.left)
      # Get max value found in left
      left_max = get_max_value(t.left, max_val)
      # Continue with right subtree
      return find_argmax(t.right, idx + left_count, left_max, left_idx)
    case Tensor/Leaf:
      if t.val > max_val:
        return idx
      else:
        return max_idx
    case Tensor/Nil:
      return max_idx

def count_leaves(t: Tensor) -> u24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return 1
    case Tensor/Nil:
      return 0

def get_max_value(t: Tensor, current_max: f24) -> f24:
  fold t:
    case Tensor/Node:
      left_max = t.left
      right_max = t.right
      if left_max > right_max:
        if left_max > current_max:
          return left_max
        else:
          return current_max
      else:
        if right_max > current_max:
          return right_max
        else:
          return current_max
    case Tensor/Leaf:
      if t.val > current_max:
        return t.val
      else:
        return current_max
    case Tensor/Nil:
      return current_max

# ============================================================
# Sample Data Generation (for testing)
# ============================================================

# Generate a synthetic MNIST-like image (for testing without real data)
# Creates a simple pattern for a digit
def generate_test_image(digit: u24, seed: u24) -> Tensor:
  # Generate 784 values with a pattern based on digit
  bend i=0, s=seed:
    when i < 784:
      # Create simple patterns for different digits
      row = i / 28
      col = i % 28
      pixel = generate_digit_pixel(digit, row, col, s)
      result = Tensor/Node {
        left: Tensor/Leaf { val: pixel },
        right: fork(i + 1, s + 1)
      }
    else:
      result = Tensor/Nil
  return result

# Generate pixel value based on digit pattern
def generate_digit_pixel(digit: u24, row: u24, col: u24, seed: u24) -> f24:
  # Simple patterns - center region brighter for each digit
  center_row = 14
  center_col = 14

  # Distance from center
  dr = if row > center_row { row - center_row } else { center_row - row }
  dc = if col > center_col { col - center_col } else { center_col - col }
  dist = dr + dc

  # Base intensity based on distance
  if dist < 5 + digit:
    # Add some randomness
    noise = lcg_pixel(seed) * 0.2
    return 200.0 + noise * 55.0
  else:
    if dist < 10 + digit:
      noise = lcg_pixel(seed + 1) * 0.3
      return 50.0 + noise * 50.0
    else:
      return lcg_pixel(seed + 2) * 20.0

def lcg_pixel(seed: u24) -> f24:
  a = 1103515245
  c = 12345
  m = 2147483648
  next_val = (a * seed + c) % m
  return next_val / m

# ============================================================
# Create Test Dataset
# ============================================================

# Create a small test dataset for MNIST training validation
def create_test_dataset(num_samples: u24, seed: u24) -> MNISTDataset:
  images = create_test_images(num_samples, seed)
  return MNISTDataset/Dataset { images: images, size: num_samples }

def create_test_images(num: u24, seed: u24) -> ImageList:
  bend i=0, s=seed:
    when i < num:
      # Cycle through digits 0-9
      digit = i % 10
      pixels = generate_test_image(digit, s)
      normalized = normalize_pixels(pixels)
      image = MNISTImage/Image { pixels: normalized, label: digit }
      result = ImageList/Cons { head: image, tail: fork(i + 1, s + 100) }
    else:
      result = ImageList/Nil
  return result

# ============================================================
# Dataset Operations
# ============================================================

# Get dataset length
def mnist_dataset_length(ds: MNISTDataset) -> u24:
  match ds:
    case MNISTDataset/Dataset:
      return ds.size
    case MNISTDataset/Empty:
      return 0

# Get image at index
def mnist_get_image(ds: MNISTDataset, idx: u24) -> MNISTImage:
  match ds:
    case MNISTDataset/Dataset:
      return image_list_get(ds.images, idx)
    case MNISTDataset/Empty:
      return MNISTImage/Empty

def image_list_get(list: ImageList, idx: u24) -> MNISTImage:
  match list:
    case ImageList/Cons:
      if idx == 0:
        return list.head
      else:
        return image_list_get(list.tail, idx - 1)
    case ImageList/Nil:
      return MNISTImage/Empty

# Extract input tensor from image
def get_input(img: MNISTImage) -> Tensor:
  match img:
    case MNISTImage/Image:
      return img.pixels
    case MNISTImage/Empty:
      return Tensor/Nil

# Extract target tensor (one-hot) from image
def get_target(img: MNISTImage) -> Tensor:
  match img:
    case MNISTImage/Image:
      return label_to_onehot(img.label)
    case MNISTImage/Empty:
      return Tensor/Nil

# Get label directly
def get_label(img: MNISTImage) -> u24:
  match img:
    case MNISTImage/Image:
      return img.label
    case MNISTImage/Empty:
      return 0

# ============================================================
# Batching for MNIST
# ============================================================

type MNISTBatch:
  Batch {
    inputs: TensorList,   # List of 784-dim input vectors
    targets: TensorList,  # List of 10-dim one-hot vectors
    labels: LabelList,    # Raw labels for accuracy computation
    size: u24
  }
  Empty

type TensorList:
  TL { head: Tensor, ~tail: TensorList }
  Nil

type LabelList:
  LL { head: u24, ~tail: LabelList }
  Nil

# Create batches from MNIST dataset
def create_mnist_batches(ds: MNISTDataset, batch_size: u24) -> MNISTBatchList:
  total = mnist_dataset_length(ds)
  return create_batches_rec(ds, batch_size, 0, total)

type MNISTBatchList:
  BL { head: MNISTBatch, ~tail: MNISTBatchList }
  Nil

def create_batches_rec(ds: MNISTDataset, batch_size: u24, start: u24, total: u24) -> MNISTBatchList:
  if start >= total:
    return MNISTBatchList/Nil
  else:
    end = start + batch_size
    if end > total:
      end = total
    batch = extract_mnist_batch(ds, start, end)
    return MNISTBatchList/BL {
      head: batch,
      tail: create_batches_rec(ds, batch_size, end, total)
    }

def extract_mnist_batch(ds: MNISTDataset, start: u24, end: u24) -> MNISTBatch:
  inputs = extract_batch_inputs(ds, start, end)
  targets = extract_batch_targets(ds, start, end)
  labels = extract_batch_labels(ds, start, end)
  return MNISTBatch/Batch {
    inputs: inputs,
    targets: targets,
    labels: labels,
    size: end - start
  }

def extract_batch_inputs(ds: MNISTDataset, start: u24, end: u24) -> TensorList:
  bend i=start:
    when i < end:
      img = mnist_get_image(ds, i)
      result = TensorList/TL {
        head: get_input(img),
        tail: fork(i + 1)
      }
    else:
      result = TensorList/Nil
  return result

def extract_batch_targets(ds: MNISTDataset, start: u24, end: u24) -> TensorList:
  bend i=start:
    when i < end:
      img = mnist_get_image(ds, i)
      result = TensorList/TL {
        head: get_target(img),
        tail: fork(i + 1)
      }
    else:
      result = TensorList/Nil
  return result

def extract_batch_labels(ds: MNISTDataset, start: u24, end: u24) -> LabelList:
  bend i=start:
    when i < end:
      img = mnist_get_image(ds, i)
      result = LabelList/LL {
        head: get_label(img),
        tail: fork(i + 1)
      }
    else:
      result = LabelList/Nil
  return result

# ============================================================
# Accuracy Computation
# ============================================================

# Compute accuracy for MNIST batch
def mnist_batch_accuracy(predictions: TensorList, labels: LabelList) -> f24:
  correct = count_mnist_correct(predictions, labels)
  total = tensor_list_length(predictions)
  if total > 0:
    return correct / total
  else:
    return 0.0

def count_mnist_correct(preds: TensorList, labels: LabelList) -> f24:
  match preds:
    case TensorList/TL:
      match labels:
        case LabelList/LL:
          pred_class = prediction_to_class(preds.head)
          is_correct = if pred_class == labels.head { 1.0 } else { 0.0 }
          rest = count_mnist_correct(preds.tail, labels.tail)
          return is_correct + rest
        case LabelList/Nil:
          return 0.0
    case TensorList/Nil:
      return 0.0

def tensor_list_length(tl: TensorList) -> f24:
  fold tl:
    case TensorList/TL:
      return 1.0 + tl.tail
    case TensorList/Nil:
      return 0.0

# ============================================================
# Data Augmentation (Optional)
# ============================================================

# Random horizontal shift (small translation)
def random_shift(pixels: Tensor, max_shift: u24, seed: u24) -> Tensor:
  # Shift amount: random in [-max_shift, max_shift]
  shift = (lcg_aug(seed) % (2 * max_shift + 1)) - max_shift
  return apply_shift(pixels, shift)

def lcg_aug(seed: u24) -> u24:
  a = 1103515245
  c = 12345
  m = 2147483648
  return (a * seed + c) % m

def apply_shift(pixels: Tensor, shift: u24) -> Tensor:
  # For simplicity, just return original (full implementation would shift pixels)
  # A full implementation would:
  # 1. Reshape 784 -> 28x28
  # 2. Shift columns by 'shift' amount
  # 3. Flatten back to 784
  return pixels

# Add random noise for robustness
def add_noise(pixels: Tensor, noise_level: f24, seed: u24) -> Tensor:
  return tree_add_noise(pixels, noise_level, seed)

def tree_add_noise(t: Tensor, noise_level: f24, seed: u24) -> Tensor:
  match t:
    case Tensor/Node:
      return Tensor/Node {
        left: tree_add_noise(t.left, noise_level, seed),
        right: tree_add_noise(t.right, noise_level, seed + 1)
      }
    case Tensor/Leaf:
      noise = (lcg_pixel(seed) - 0.5) * noise_level
      new_val = t.val + noise
      # Clamp to [0, 1]
      if new_val < 0.0:
        return Tensor/Leaf { val: 0.0 }
      else:
        if new_val > 1.0:
          return Tensor/Leaf { val: 1.0 }
        else:
          return Tensor/Leaf { val: new_val }
    case Tensor/Nil:
      return Tensor/Nil

# ============================================================
# Confusion Matrix
# ============================================================

# For 10-class MNIST, confusion matrix is 10x10
type ConfusionMatrix:
  CM { data: Tensor }  # 100 elements (10x10 flattened)
  Empty

def init_confusion_matrix() -> ConfusionMatrix:
  # 10x10 = 100 zeros
  data = create_zeros(100)
  return ConfusionMatrix/CM { data: data }

def create_zeros(size: u24) -> Tensor:
  bend i=0:
    when i < size:
      result = Tensor/Node {
        left: Tensor/Leaf { val: 0.0 },
        right: fork(i + 1)
      }
    else:
      result = Tensor/Nil
  return result

# Update confusion matrix with a prediction
def update_confusion_matrix(cm: ConfusionMatrix, true_label: u24, pred_label: u24) -> ConfusionMatrix:
  match cm:
    case ConfusionMatrix/CM:
      # Index = true_label * 10 + pred_label
      idx = true_label * 10 + pred_label
      new_data = increment_at(cm.data, idx, 0)
      return ConfusionMatrix/CM { data: new_data }
    case ConfusionMatrix/Empty:
      return ConfusionMatrix/Empty

def increment_at(t: Tensor, target_idx: u24, current_idx: u24) -> Tensor:
  match t:
    case Tensor/Node:
      left_count = count_leaves(t.left)
      if target_idx < current_idx + left_count:
        return Tensor/Node {
          left: increment_at(t.left, target_idx, current_idx),
          right: t.right
        }
      else:
        return Tensor/Node {
          left: t.left,
          right: increment_at(t.right, target_idx, current_idx + left_count)
        }
    case Tensor/Leaf:
      if current_idx == target_idx:
        return Tensor/Leaf { val: t.val + 1.0 }
      else:
        return t
    case Tensor/Nil:
      return Tensor/Nil
