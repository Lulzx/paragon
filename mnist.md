# MNIST Classification in Bend

A high-performance MNIST-style digit classifier implemented in Bend, achieving **99.6% accuracy** using fully parallel tree operations.

## Overview

This implementation demonstrates how to build neural networks that work *with* Bend's tree-based computational model rather than against it. The key insight is that parallel fold operations over aligned tree structures can replace traditional matrix multiplication.

## Architecture

```
Input (16 features) → Linear Layer → Softmax → Output (10 classes)
```

- **Input**: 16-element tree with digit-specific activation patterns
- **Weights**: Tree of weight vectors (WeightTree), one per output class
- **Output**: 10-class probability distribution via softmax

## Key Design Principles

### 1. Aligned Tree Structures

The weight structure mirrors the output structure exactly:

```bend
type WeightTree:
  WNode { ~left: WeightTree, ~right: WeightTree }
  WLeaf { weights: Tensor }  # Weight vector for one class
  WNil
```

This alignment enables parallel gradient updates - the gradient tensor and weight tree can be zipped together in O(log n) operations.

### 2. Parallel Fold as Matrix Multiplication

Instead of sequential index-based access:
```python
# Traditional (O(n) sequential)
for i in range(classes):
    score[i] = dot(weights[i], input)
```

We use parallel fold:
```bend
# Bend (O(log n) parallel)
def compute_all_scores(wt: WeightTree, input: Tensor) -> Tensor:
  fold wt with input:
    case WeightTree/WNode:
      return Tensor/Node {
        left: wt.left(input),
        right: wt.right(input)
      }
    case WeightTree/WLeaf:
      score = tree_dot(wt.weights, input)
      return Tensor/Leaf { val: score }
```

### 3. Parallel Gradient Updates

The backward pass exploits structural alignment:

```bend
def update_weight_tree(wt: WeightTree, d_out: Tensor, input: Tensor, lr: f24) -> WeightTree:
  match wt:
    case WeightTree/WNode:
      match d_out:
        case Tensor/Node:
          return WeightTree/WNode {
            left: update_weight_tree(wt.left, d_out.left, input, lr),
            right: update_weight_tree(wt.right, d_out.right, input, lr)
          }
    case WeightTree/WLeaf:
      match d_out:
        case Tensor/Leaf:
          grad = tree_scale(input, d_out.val)
          new_weights = tree_sub(wt.weights, tree_scale(grad, lr))
          return WeightTree/WLeaf { weights: new_weights }
```

## Performance

| Metric | Value |
|--------|-------|
| Training samples | 5,000 (500 per digit) |
| Test samples | 500 |
| Accuracy | 99.6% (498/500) |
| Learning rate | 0.2 |

## Running

```bash
bend run examples/mnist_parallel.bend
```

Output: `Result: 99.600` (accuracy percentage)

## Files

- `examples/mnist_parallel.bend` - Main implementation with parallel tree operations
- `examples/mnist_tree_native.bend` - Earlier iteration using native tree ops
- `examples/mnist_simple_correct.bend` - Simple linear classifier baseline

## Lessons Learned

1. **Work with the model**: Bend's tree structure isn't a limitation - it's a feature. Align your data structures to enable parallel operations.

2. **Avoid O(n) indexing**: Sequential tree traversal kills parallelism. Structure data so operations can proceed via fold/match.

3. **Structure alignment matters**: When two trees have the same shape, you can zip them together for parallel element-wise operations.

4. **Fold is your friend**: The `fold` construct enables O(log n) parallel reductions - use it for sums, dot products, and score computations.
