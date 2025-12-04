# Paragon Benchmarks

Performance benchmarks for HVM3-based neural network operations and parallel algorithms.

## Test Environment

- **Platform**: macOS (Darwin)
- **HVM3 Version**: 0.1.0.0
- **Mode**: Compiled (-c flag)

## Benchmark Results

### 1. Parallel Sum (Tree Reduction)

Demonstrates O(log n) parallel time complexity for summing values.

| Dataset Size | Interactions | Time (s) | Memory (nodes) | Performance (MIPS) |
|-------------|-------------|----------|----------------|-------------------|
| 65,536 values | 1,310,708 | 0.0017 | 524,284 | **773 MIPS** |
| 1,048,576 values | 20,971,508 | 0.0279 | 8,388,604 | **752 MIPS** |

### 2. Fibonacci (Recursive)

Demonstrates parallel recursive computation with exponential branching.

| Input | Result | Interactions | Time (s) | Performance (MIPS) |
|-------|--------|-------------|----------|-------------------|
| fib(20) | 6765 | 2,189 | 0.000044 | **50 MIPS** |

### 3. Bitonic Sort (Parallel Sorting)

Demonstrates parallel sorting with O(log² n) parallel time complexity.

| Elements | Interactions | Time (s) | Memory (nodes) | Performance (MIPS) |
|----------|-------------|----------|----------------|-------------------|
| 8 | 586 | 0.000039 | 136 | **15 MIPS** |

### 4. Matrix-Vector Multiplication

Demonstrates parallel dot products and matrix operations.

| Matrix Size | Interactions | Time (s) | Memory (nodes) | Performance (MIPS) |
|------------|-------------|----------|----------------|-------------------|
| 8×8 | 2,797 | 0.000059 | 971 | **47 MIPS** |

### 5. Neural Network Layer Forward Pass

Demonstrates parallel neural network computation with ReLU activation.

| Neurons | Interactions | Time (s) | Memory (nodes) | Performance (MIPS) |
|---------|-------------|----------|----------------|-------------------|
| 16 | 1,602 | 0.000038 | 298 | **42 MIPS** |

### 6. XOR Training Example

Complete neural network training loop with forward pass, loss computation, and epochs.

| Epochs | Interactions | Time (s) | Memory (nodes) | Performance (MIPS) |
|--------|-------------|----------|----------------|-------------------|
| 10 | 690 | 0.000040 | 1,126 | **17 MIPS** |

## Interpreted vs Compiled Mode

HVM3's compiled mode (-c flag) provides significant speedup by generating optimized C code.

| Mode | Parallel Sum (65k) | Speedup |
|------|-------------------|---------|
| Interpreted | 6.6 MIPS | 1× |
| Compiled | **777 MIPS** | **118×** |

## Key Observations

### Performance Characteristics

1. **Parallel Reduction Scales Well**: Tree-based parallel sum achieves ~750 MIPS consistently across different dataset sizes, demonstrating efficient parallel reduction.

2. **Compiled Mode Essential**: The compiled mode provides ~100× speedup over interpreted mode, essential for production workloads.

3. **Linear Types Overhead**: HVM3's linear/affine type system requires explicit duplication (`!&{...}=`) which adds some overhead but enables safe parallelism.

4. **Memory Efficiency**: The interaction calculus model provides efficient memory usage through automatic garbage collection of affine terms.

### Parallelism Model

HVM3 achieves parallelism through:

- **Tree-based data structures**: Enable O(log n) parallel reductions
- **First-class duplications**: Allow safe copying for parallel branches
- **Optimal β-reduction**: Avoids redundant computation through sharing

### Comparison with Traditional Approaches

| Operation | Traditional (single-core) | HVM3 (parallel) |
|-----------|--------------------------|-----------------|
| Tree Sum (1M) | O(n) sequential | O(log n) parallel depth |
| Bitonic Sort | O(n log² n) sequential | O(log² n) parallel depth |
| Matrix-Vector | O(n²) sequential | O(log n) parallel depth |

## Running Benchmarks

```bash
# Run individual benchmarks
hvm run examples/bench_fibonacci.hvm -c -s
hvm run examples/bench_parallel_sum.hvm -c -s
hvm run examples/bench_bitonic_sort.hvm -c -s
hvm run examples/bench_matmul.hvm -c -s
hvm run examples/bench_neural_layer.hvm -c -s
hvm run examples/training_example.hvm -c -s

# Flags:
#   -c  Compiled mode (faster)
#   -s  Show statistics
```

## Apple Silicon GPU Backend (Metal)

Paragon includes a native Metal backend for Apple Silicon Macs, providing GPU-accelerated neural network operations.

### Metal Shader Performance

The Metal backend implements the following GPU kernels:

| Kernel | Description | Optimization |
|--------|-------------|--------------|
| `parallel_sum_reduce` | Tree-based parallel sum | Shared memory reduction |
| `tensor_add/mul/relu` | Element-wise operations | Coalesced memory access |
| `dense_forward` | Layer forward pass | Tiled matrix multiplication |
| `dense_weight_gradients` | Weight gradient computation | 2D thread dispatch |
| `dense_input_gradients` | Backprop to previous layer | Transpose multiply |
| `sgd_update` | Weight updates with momentum | In-place update |
| `mse_loss` | Loss computation | Parallel reduction |
| `bitonic_sort_step` | Parallel sorting | Compare-swap network |

### Expected GPU Performance

Based on Apple Silicon GPU specifications:

| Device | GPU Cores | Memory Bandwidth | Est. Performance |
|--------|-----------|------------------|------------------|
| M1 | 8 | 68 GB/s | ~5,000 MIPS |
| M1 Pro | 16 | 200 GB/s | ~15,000 MIPS |
| M1 Max | 32 | 400 GB/s | ~30,000 MIPS |
| M2 | 10 | 100 GB/s | ~8,000 MIPS |
| M3 Max | 40 | 400 GB/s | ~40,000 MIPS |
| M4 Max | 40 | 546 GB/s | ~50,000 MIPS |

### Building and Testing Metal Backend

```bash
# Build Metal shaders and Swift runtime
cd metal
./build_metal.sh

# Verify Metal support
./build/test_metal
```

### Unified Memory Advantage

Apple Silicon's unified memory architecture provides:
- Zero-copy data transfer between CPU and GPU
- Efficient memory sharing for HVM3's tree structures
- Reduced latency for interactive training

## Parallel Backpropagation

The `examples/backprop.hvm` implements full parallel backpropagation in HVM3:

### Gradient Computation Complexity

| Operation | Sequential | Parallel (HVM3) |
|-----------|------------|-----------------|
| Forward pass | O(n) | O(log n) depth |
| Loss gradient | O(n) | O(1) |
| Weight gradients | O(n²) | O(log n) depth |
| Input gradients | O(n²) | O(log n) depth |
| Weight updates | O(n) | O(1) |

### Running Backprop Example

```bash
hvm run examples/backprop.hvm -c -s
```

## Future Optimizations

1. ~~**GPU Backend**: HVM3 can target CUDA for massive parallelism~~ ✓ Metal backend implemented
2. **Larger Networks**: Scale to deeper/wider neural networks
3. **Batched Training**: Process multiple samples in parallel
4. ~~**Gradient Computation**: Implement parallel backpropagation~~ ✓ Implemented
5. **CUDA Backend**: Port Metal shaders to CUDA for NVIDIA GPUs
6. **Distributed Training**: Multi-device gradient synchronization
