# Matrix operations library for Paragon
# Provides parallel matrix operations for neural network computations

# Matrix represented as a tree structure for parallel processing
type Matrix:
  Mat { rows: u24, cols: u24, data: Tensor }
  Empty

# Tree structure for storing matrix data (enables parallel operations)
type Tensor:
  Node { ~left: Tensor, ~right: Tensor }
  Leaf { val: f24 }
  Nil

# Vector type for 1D operations
type Vector:
  Vec { len: u24, data: Tensor }
  Empty

# ============================================================
# Matrix Creation
# ============================================================

# Create a matrix filled with zeros
def mat_zeros(rows: u24, cols: u24) -> Matrix:
  bend r=0:
    when r < rows:
      row = fork(r + 1)
    else:
      row = Tensor/Nil
  return Matrix/Mat { rows: rows, cols: cols, data: row }

# Create a vector of given length with initial value
def vec_create(len: u24, init: f24) -> Vector:
  bend i=0:
    when i < len:
      data = Tensor/Node { left: Tensor/Leaf { val: init }, right: fork(i + 1) }
    else:
      data = Tensor/Nil
  return Vector/Vec { len: len, data: data }

# ============================================================
# Element-wise Operations (Parallel)
# ============================================================

# Add two trees element-wise (parallel)
def tree_add(a: Tensor, b: Tensor) -> Tensor:
  match a:
    case Tensor/Node:
      match b:
        case Tensor/Node:
          return Tensor/Node {
            left: tree_add(a.left, b.left),
            right: tree_add(a.right, b.right)
          }
        case Tensor/Leaf:
          return a
        case Tensor/Nil:
          return a
    case Tensor/Leaf:
      match b:
        case Tensor/Leaf:
          return Tensor/Leaf { val: a.val + b.val }
        case _:
          return a
    case Tensor/Nil:
      return b

# Multiply two trees element-wise (parallel)
def tree_mul(a: Tensor, b: Tensor) -> Tensor:
  match a:
    case Tensor/Node:
      match b:
        case Tensor/Node:
          return Tensor/Node {
            left: tree_mul(a.left, b.left),
            right: tree_mul(a.right, b.right)
          }
        case Tensor/Leaf:
          return a
        case Tensor/Nil:
          return Tensor/Nil
    case Tensor/Leaf:
      match b:
        case Tensor/Leaf:
          return Tensor/Leaf { val: a.val * b.val }
        case _:
          return Tensor/Nil
    case Tensor/Nil:
      return Tensor/Nil

# Scale tree by a constant (parallel)
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

# Sum all elements in a tree (parallel reduction)
def tree_sum(t: Tensor) -> f24:
  fold t:
    case Tensor/Node:
      return t.left + t.right
    case Tensor/Leaf:
      return t.val
    case Tensor/Nil:
      return 0.0

# ============================================================
# Matrix Operations
# ============================================================

# Matrix addition (parallel)
def mat_add(a: Matrix, b: Matrix) -> Matrix:
  match a:
    case Matrix/Mat:
      match b:
        case Matrix/Mat:
          return Matrix/Mat {
            rows: a.rows,
            cols: a.cols,
            data: tree_add(a.data, b.data)
          }
        case Matrix/Empty:
          return a
    case Matrix/Empty:
      return b

# Matrix scalar multiplication (parallel)
def mat_scale(m: Matrix, s: f24) -> Matrix:
  match m:
    case Matrix/Mat:
      return Matrix/Mat {
        rows: m.rows,
        cols: m.cols,
        data: tree_scale(m.data, s)
      }
    case Matrix/Empty:
      return Matrix/Empty

# Vector dot product (parallel)
def vec_dot(a: Vector, b: Vector) -> f24:
  match a:
    case Vector/Vec:
      match b:
        case Vector/Vec:
          return tree_sum(tree_mul(a.data, b.data))
        case Vector/Empty:
          return 0.0
    case Vector/Empty:
      return 0.0

# Vector addition (parallel)
def vec_add(a: Vector, b: Vector) -> Vector:
  match a:
    case Vector/Vec:
      match b:
        case Vector/Vec:
          return Vector/Vec {
            len: a.len,
            data: tree_add(a.data, b.data)
          }
        case Vector/Empty:
          return a
    case Vector/Empty:
      return b

# Vector scaling (parallel)
def vec_scale(v: Vector, s: f24) -> Vector:
  match v:
    case Vector/Vec:
      return Vector/Vec {
        len: v.len,
        data: tree_scale(v.data, s)
      }
    case Vector/Empty:
      return Vector/Empty
