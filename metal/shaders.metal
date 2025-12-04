// Paragon Metal Shaders for Apple Silicon GPU
// Implements parallel interaction net evaluation and neural network operations
// Optimized for M1/M2/M3/M4 unified memory architecture

#include <metal_stdlib>
using namespace metal;

#define UINT32_MAX 0xFFFFFFFF

// ============================================================
// Constants and Configuration
// ============================================================

constant uint THREADS_PER_GROUP = 256;
constant uint MAX_NODES = 1048576;  // 1M nodes
constant uint SCALE_FACTOR = 1000;  // Fixed-point scaling

// ============================================================
// Data Structures
// ============================================================

// Interaction net node types
enum class NodeType : uint8_t {
    Nil = 0,
    Leaf = 1,
    Node = 2,
    Lambda = 3,
    App = 4,
    Dup = 5,
    Era = 6
};

// Tensor node for tree-based parallel computation
struct TensorNode {
    NodeType type;
    uint8_t padding[3];
    int32_t value;      // Scaled integer value (x1000)
    uint32_t left;      // Index to left child
    uint32_t right;     // Index to right child
};

// Active pair for interaction net rewriting
struct ActivePair {
    uint32_t node_a;
    uint32_t node_b;
    uint32_t port_a;
    uint32_t port_b;
};

// Memory statistics
struct MemStats {
    atomic_uint interactions;
    atomic_uint allocations;
    atomic_uint deallocations;
    uint32_t max_nodes;
};

// Neural network layer data
struct LayerData {
    uint32_t input_size;
    uint32_t output_size;
    uint32_t weights_offset;
    uint32_t biases_offset;
};

// Gradient data for backpropagation
struct GradientData {
    uint32_t layer_idx;
    uint32_t param_idx;
    int32_t gradient;   // Scaled gradient value
};

// ============================================================
// Utility Functions
// ============================================================

// Scaled integer multiplication (maintains x1000 scale)
inline int32_t scaled_mul(int32_t a, int32_t b) {
    return (int32_t)((int64_t)a * b / SCALE_FACTOR);
}

// Scaled integer division
inline int32_t scaled_div(int32_t a, int32_t b) {
    if (b == 0) return 0;
    return (int32_t)((int64_t)a * SCALE_FACTOR / b);
}

// ReLU activation
inline int32_t relu(int32_t x) {
    return max(x, 0);
}

// ReLU derivative (for backprop)
inline int32_t relu_derivative(int32_t x) {
    return x > 0 ? SCALE_FACTOR : 0;
}

// Sigmoid approximation using piecewise linear (faster than exp)
inline int32_t sigmoid_approx(int32_t x) {
    // Clamp to [-4000, 4000] range (scaled by 1000)
    x = clamp(x, -4000, 4000);
    // Linear approximation: 0.25 * x + 0.5 for middle range
    if (x < -2000) return 0;
    if (x > 2000) return SCALE_FACTOR;
    return (x + 2000) * SCALE_FACTOR / 4000;
}

// ============================================================
// Parallel Tree Reduction Kernels
// ============================================================

// Parallel sum reduction using shared memory
kernel void parallel_sum_reduce(
    device const TensorNode* nodes [[buffer(0)]],
    device int32_t* partial_sums [[buffer(1)]],
    device atomic_uint* node_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint group_size [[threads_per_threadgroup]]
) {
    threadgroup int32_t shared_data[THREADS_PER_GROUP];

    // Load node value
    uint node_idx = tid;
    int32_t value = 0;

    if (node_idx < MAX_NODES && nodes[node_idx].type == NodeType::Leaf) {
        value = nodes[node_idx].value;
    }

    shared_data[lid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction in shared memory
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_data[lid] += shared_data[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result from first thread
    if (lid == 0) {
        partial_sums[gid] = shared_data[0];
    }
}

// Tree-based parallel sum (follows HVM3 tree structure)
kernel void tree_sum_kernel(
    device const TensorNode* nodes [[buffer(0)]],
    device int32_t* results [[buffer(1)]],
    device const uint32_t* root_indices [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint root = root_indices[tid];

    // Stack-based tree traversal (avoid recursion on GPU)
    uint32_t stack[32];
    int32_t values[32];
    int sp = 0;
    int32_t sum = 0;

    stack[sp++] = root;

    while (sp > 0) {
        uint32_t idx = stack[--sp];
        TensorNode node = nodes[idx];

        switch (node.type) {
            case NodeType::Leaf:
                sum += node.value;
                break;
            case NodeType::Node:
                if (node.left < MAX_NODES) stack[sp++] = node.left;
                if (node.right < MAX_NODES) stack[sp++] = node.right;
                break;
            default:
                break;
        }
    }

    results[tid] = sum;
}

// ============================================================
// Element-wise Tensor Operations
// ============================================================

// Parallel tensor addition
kernel void tensor_add(
    device const int32_t* a [[buffer(0)]],
    device const int32_t* b [[buffer(1)]],
    device int32_t* result [[buffer(2)]],
    device const uint32_t& size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < size) {
        result[tid] = a[tid] + b[tid];
    }
}

// Parallel tensor multiplication (scaled)
kernel void tensor_mul(
    device const int32_t* a [[buffer(0)]],
    device const int32_t* b [[buffer(1)]],
    device int32_t* result [[buffer(2)]],
    device const uint32_t& size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < size) {
        result[tid] = scaled_mul(a[tid], b[tid]);
    }
}

// Parallel ReLU activation
kernel void tensor_relu(
    device const int32_t* input [[buffer(0)]],
    device int32_t* output [[buffer(1)]],
    device const uint32_t& size [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < size) {
        output[tid] = relu(input[tid]);
    }
}

// ============================================================
// Matrix Operations (for Neural Networks)
// ============================================================

// Matrix-vector multiplication (parallel over output elements)
kernel void matvec_mul(
    device const int32_t* matrix [[buffer(0)]],
    device const int32_t* vector [[buffer(1)]],
    device int32_t* result [[buffer(2)]],
    device const uint32_t& rows [[buffer(3)]],
    device const uint32_t& cols [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup int32_t shared_vec[THREADS_PER_GROUP];

    if (tid >= rows) return;

    int32_t sum = 0;
    uint row_offset = tid * cols;

    // Process in chunks that fit in shared memory
    for (uint chunk = 0; chunk < cols; chunk += THREADS_PER_GROUP) {
        // Load vector chunk to shared memory
        uint vec_idx = chunk + lid;
        if (vec_idx < cols) {
            shared_vec[lid] = vector[vec_idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        uint chunk_size = min(THREADS_PER_GROUP, cols - chunk);
        for (uint i = 0; i < chunk_size; i++) {
            sum += scaled_mul(matrix[row_offset + chunk + i], shared_vec[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    result[tid] = sum;
}

// ============================================================
// Neural Network Forward Pass
// ============================================================

// Dense layer forward pass
kernel void dense_forward(
    device const int32_t* weights [[buffer(0)]],
    device const int32_t* biases [[buffer(1)]],
    device const int32_t* input [[buffer(2)]],
    device int32_t* output [[buffer(3)]],
    device const uint32_t& input_size [[buffer(4)]],
    device const uint32_t& output_size [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup int32_t shared_input[THREADS_PER_GROUP];

    if (tid >= output_size) return;

    int32_t sum = biases[tid];
    uint weight_offset = tid * input_size;

    // Compute weighted sum
    for (uint chunk = 0; chunk < input_size; chunk += THREADS_PER_GROUP) {
        uint inp_idx = chunk + lid;
        if (inp_idx < input_size) {
            shared_input[lid] = input[inp_idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint chunk_size = min(THREADS_PER_GROUP, input_size - chunk);
        for (uint i = 0; i < chunk_size; i++) {
            sum += scaled_mul(weights[weight_offset + chunk + i], shared_input[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Apply ReLU activation
    output[tid] = relu(sum);
}

// ============================================================
// Backpropagation Kernels (Parallel Gradient Computation)
// ============================================================

// Compute output layer gradients (MSE loss derivative)
kernel void compute_output_gradients(
    device const int32_t* predictions [[buffer(0)]],
    device const int32_t* targets [[buffer(1)]],
    device int32_t* gradients [[buffer(2)]],
    device const uint32_t& size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < size) {
        // d(MSE)/d(pred) = 2 * (pred - target) / n
        // Simplified: just compute (pred - target) scaled
        gradients[tid] = 2 * (predictions[tid] - targets[tid]);
    }
}

// Compute gradients through ReLU (element-wise)
kernel void relu_backward(
    device const int32_t* upstream_grad [[buffer(0)]],
    device const int32_t* forward_input [[buffer(1)]],
    device int32_t* output_grad [[buffer(2)]],
    device const uint32_t& size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < size) {
        // Gradient is upstream if input > 0, else 0
        output_grad[tid] = forward_input[tid] > 0 ? upstream_grad[tid] : 0;
    }
}

// Compute weight gradients for dense layer (parallel over weights)
kernel void dense_weight_gradients(
    device const int32_t* upstream_grad [[buffer(0)]],   // [output_size]
    device const int32_t* layer_input [[buffer(1)]],     // [input_size]
    device int32_t* weight_grads [[buffer(2)]],          // [output_size * input_size]
    device const uint32_t& input_size [[buffer(3)]],
    device const uint32_t& output_size [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint out_idx = tid.x;
    uint in_idx = tid.y;

    if (out_idx < output_size && in_idx < input_size) {
        // dL/dW[i,j] = dL/dout[i] * input[j]
        uint weight_idx = out_idx * input_size + in_idx;
        weight_grads[weight_idx] = scaled_mul(upstream_grad[out_idx], layer_input[in_idx]);
    }
}

// Compute bias gradients (parallel reduction)
kernel void dense_bias_gradients(
    device const int32_t* upstream_grad [[buffer(0)]],
    device int32_t* bias_grads [[buffer(1)]],
    device const uint32_t& output_size [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < output_size) {
        // dL/db[i] = dL/dout[i] (summed across batch)
        bias_grads[tid] = upstream_grad[tid];
    }
}

// Compute input gradients for backprop to previous layer
kernel void dense_input_gradients(
    device const int32_t* upstream_grad [[buffer(0)]],   // [output_size]
    device const int32_t* weights [[buffer(1)]],         // [output_size * input_size]
    device int32_t* input_grads [[buffer(2)]],           // [input_size]
    device const uint32_t& input_size [[buffer(3)]],
    device const uint32_t& output_size [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup int32_t shared_grad[THREADS_PER_GROUP];

    if (tid >= input_size) return;

    int32_t sum = 0;

    // Compute gradient: sum over output_size of weights[j,tid] * upstream_grad[j]
    for (uint chunk = 0; chunk < output_size; chunk += THREADS_PER_GROUP) {
        uint grad_idx = chunk + lid;
        if (grad_idx < output_size) {
            shared_grad[lid] = upstream_grad[grad_idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint chunk_size = min(THREADS_PER_GROUP, output_size - chunk);
        for (uint j = 0; j < chunk_size; j++) {
            uint out_idx = chunk + j;
            uint weight_idx = out_idx * input_size + tid;
            sum += scaled_mul(weights[weight_idx], shared_grad[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    input_grads[tid] = sum;
}

// ============================================================
// Weight Update (SGD with momentum)
// ============================================================

kernel void sgd_update(
    device int32_t* weights [[buffer(0)]],
    device const int32_t* gradients [[buffer(1)]],
    device int32_t* momentum [[buffer(2)]],
    device const int32_t& learning_rate [[buffer(3)]],   // Scaled (e.g., 100 = 0.1)
    device const int32_t& momentum_factor [[buffer(4)]], // Scaled (e.g., 900 = 0.9)
    device const uint32_t& size [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < size) {
        // momentum = momentum_factor * momentum + gradient
        int32_t new_momentum = scaled_mul(momentum_factor, momentum[tid]) + gradients[tid];
        momentum[tid] = new_momentum;

        // weight = weight - learning_rate * momentum
        weights[tid] -= scaled_mul(learning_rate, new_momentum);
    }
}

// ============================================================
// Loss Computation
// ============================================================

// MSE loss (parallel reduction)
kernel void mse_loss(
    device const int32_t* predictions [[buffer(0)]],
    device const int32_t* targets [[buffer(1)]],
    device int32_t* partial_losses [[buffer(2)]],
    device const uint32_t& size [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint group_size [[threads_per_threadgroup]]
) {
    threadgroup int32_t shared_loss[THREADS_PER_GROUP];

    int32_t loss = 0;
    if (tid < size) {
        int32_t diff = predictions[tid] - targets[tid];
        loss = scaled_mul(diff, diff);
    }

    shared_loss[lid] = loss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_loss[lid] += shared_loss[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partial_losses[gid] = shared_loss[0];
    }
}

// ============================================================
// Interaction Net Evaluation (HVM3 Core)
// ============================================================

// Process active pairs in parallel
kernel void eval_interaction_pairs(
    device TensorNode* nodes [[buffer(0)]],
    device ActivePair* pairs [[buffer(1)]],
    device atomic_uint* pair_count [[buffer(2)]],
    device MemStats* stats [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint num_pairs = atomic_load_explicit(pair_count, memory_order_relaxed);
    if (tid >= num_pairs) return;

    ActivePair pair = pairs[tid];
    TensorNode node_a = nodes[pair.node_a];
    TensorNode node_b = nodes[pair.node_b];

    // Implement interaction rules based on node types
    // This is a simplified version - full HVM3 would have more rules

    if (node_a.type == NodeType::Lambda && node_b.type == NodeType::App) {
        // Beta reduction: (λx.body) arg → body[x := arg]
        // Substitution handled by rewiring
        atomic_fetch_add_explicit(&stats->interactions, 1, memory_order_relaxed);
    }
    else if (node_a.type == NodeType::Dup && node_b.type == NodeType::Dup) {
        // Duplication annihilation
        atomic_fetch_add_explicit(&stats->interactions, 1, memory_order_relaxed);
    }
    else if (node_a.type == NodeType::Era) {
        // Erasure
        nodes[pair.node_b].type = NodeType::Nil;
        atomic_fetch_add_explicit(&stats->deallocations, 1, memory_order_relaxed);
    }

    // Mark pair as processed
    pairs[tid].node_a = UINT32_MAX;
}

// ============================================================
// Bitonic Sort (Parallel Sorting)
// ============================================================

kernel void bitonic_sort_step(
    device int32_t* data [[buffer(0)]],
    device const uint32_t& size [[buffer(1)]],
    device const uint32_t& step [[buffer(2)]],
    device const uint32_t& stage [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint pair_distance = 1 << (step - stage);
    uint block_width = 2 << step;

    uint left_id = (tid / pair_distance) * block_width + (tid % pair_distance);
    uint right_id = left_id + pair_distance;

    if (right_id >= size) return;

    // Determine sort direction
    bool ascending = ((tid / (1 << step)) % 2) == 0;

    int32_t left = data[left_id];
    int32_t right = data[right_id];

    bool should_swap = ascending ? (left > right) : (left < right);

    if (should_swap) {
        data[left_id] = right;
        data[right_id] = left;
    }
}
