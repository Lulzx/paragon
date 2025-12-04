// Paragon Metal Runtime for Apple Silicon
// Swift wrapper for GPU-accelerated neural network operations
// Optimized for M1/M2/M3/M4 unified memory architecture

import Metal
import MetalPerformanceShaders
import Foundation

// MARK: - Constants

let SCALE_FACTOR: Int32 = 1000
let MAX_NODES: Int = 1_048_576
let THREADS_PER_GROUP: Int = 256

// MARK: - Error Types

enum ParagonError: Error {
    case metalNotSupported
    case deviceNotFound
    case libraryLoadFailed
    case functionNotFound(String)
    case bufferAllocationFailed
    case commandEncodingFailed
    case computeFailed
}

// MARK: - Tensor Structure

struct Tensor {
    var data: [Int32]
    var shape: [Int]

    init(shape: [Int], fill: Int32 = 0) {
        self.shape = shape
        let size = shape.reduce(1, *)
        self.data = [Int32](repeating: fill, count: size)
    }

    init(data: [Int32], shape: [Int]) {
        self.data = data
        self.shape = shape
    }

    var count: Int { data.count }

    subscript(indices: Int...) -> Int32 {
        get {
            var idx = 0
            var stride = 1
            for i in (0..<indices.count).reversed() {
                idx += indices[i] * stride
                stride *= shape[i]
            }
            return data[idx]
        }
        set {
            var idx = 0
            var stride = 1
            for i in (0..<indices.count).reversed() {
                idx += indices[i] * stride
                stride *= shape[i]
            }
            data[idx] = newValue
        }
    }
}

// MARK: - Layer Structures

struct DenseLayer {
    var weights: Tensor
    var biases: Tensor
    var weightMomentum: Tensor
    var biasMomentum: Tensor

    init(inputSize: Int, outputSize: Int, seed: UInt64 = 42) {
        // Xavier initialization (scaled integers)
        let scale = Int32(sqrt(2.0 / Double(inputSize)) * Double(SCALE_FACTOR))
        var rng = SeededRNG(seed: seed)

        var weightData = [Int32](repeating: 0, count: inputSize * outputSize)
        for i in 0..<weightData.count {
            weightData[i] = rng.nextScaled(scale: scale)
        }

        self.weights = Tensor(data: weightData, shape: [outputSize, inputSize])
        self.biases = Tensor(shape: [outputSize], fill: 0)
        self.weightMomentum = Tensor(shape: [outputSize, inputSize], fill: 0)
        self.biasMomentum = Tensor(shape: [outputSize], fill: 0)
    }
}

// Simple seeded RNG for reproducible initialization
struct SeededRNG {
    var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }

    mutating func nextScaled(scale: Int32) -> Int32 {
        let val = next()
        let normalized = Double(val % 10000) / 10000.0 - 0.5
        return Int32(normalized * Double(scale) * 2)
    }
}

// MARK: - Metal Runtime

class ParagonMetal {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary

    // Compute pipelines
    var tensorAddPipeline: MTLComputePipelineState?
    var tensorMulPipeline: MTLComputePipelineState?
    var tensorReluPipeline: MTLComputePipelineState?
    var denseForwardPipeline: MTLComputePipelineState?
    var mseLossPipeline: MTLComputePipelineState?
    var outputGradientsPipeline: MTLComputePipelineState?
    var reluBackwardPipeline: MTLComputePipelineState?
    var weightGradientsPipeline: MTLComputePipelineState?
    var biasGradientsPipeline: MTLComputePipelineState?
    var inputGradientsPipeline: MTLComputePipelineState?
    var sgdUpdatePipeline: MTLComputePipelineState?
    var parallelSumPipeline: MTLComputePipelineState?
    var treeSumPipeline: MTLComputePipelineState?
    var bitonicSortPipeline: MTLComputePipelineState?

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw ParagonError.deviceNotFound
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw ParagonError.metalNotSupported
        }
        self.commandQueue = queue

        // Load shader library
        let libraryURL = Bundle.main.url(forResource: "shaders", withExtension: "metallib")
            ?? URL(fileURLWithPath: "metal/shaders.metallib")

        if FileManager.default.fileExists(atPath: libraryURL.path) {
            self.library = try device.makeLibrary(URL: libraryURL)
        } else {
            // Compile from source
            let sourceURL = URL(fileURLWithPath: "metal/shaders.metal")
            let source = try String(contentsOf: sourceURL)
            self.library = try device.makeLibrary(source: source, options: nil)
        }

        try setupPipelines()
    }

    private func setupPipelines() throws {
        tensorAddPipeline = try makePipeline("tensor_add")
        tensorMulPipeline = try makePipeline("tensor_mul")
        tensorReluPipeline = try makePipeline("tensor_relu")
        denseForwardPipeline = try makePipeline("dense_forward")
        mseLossPipeline = try makePipeline("mse_loss")
        outputGradientsPipeline = try makePipeline("compute_output_gradients")
        reluBackwardPipeline = try makePipeline("relu_backward")
        weightGradientsPipeline = try makePipeline("dense_weight_gradients")
        biasGradientsPipeline = try makePipeline("dense_bias_gradients")
        inputGradientsPipeline = try makePipeline("dense_input_gradients")
        sgdUpdatePipeline = try makePipeline("sgd_update")
        parallelSumPipeline = try makePipeline("parallel_sum_reduce")
        treeSumPipeline = try makePipeline("tree_sum_kernel")
        bitonicSortPipeline = try makePipeline("bitonic_sort_step")
    }

    private func makePipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            throw ParagonError.functionNotFound(name)
        }
        return try device.makeComputePipelineState(function: function)
    }

    // MARK: - Buffer Management

    func makeBuffer<T>(_ data: [T]) -> MTLBuffer? {
        return device.makeBuffer(bytes: data,
                                length: MemoryLayout<T>.stride * data.count,
                                options: .storageModeShared)
    }

    func makeBuffer<T>(count: Int, type: T.Type) -> MTLBuffer? {
        return device.makeBuffer(length: MemoryLayout<T>.stride * count,
                                options: .storageModeShared)
    }

    // MARK: - Forward Pass

    func denseForward(layer: DenseLayer, input: Tensor) throws -> Tensor {
        let inputSize = UInt32(layer.weights.shape[1])
        let outputSize = UInt32(layer.weights.shape[0])

        guard let weightsBuffer = makeBuffer(layer.weights.data),
              let biasesBuffer = makeBuffer(layer.biases.data),
              let inputBuffer = makeBuffer(input.data),
              let outputBuffer = makeBuffer(count: Int(outputSize), type: Int32.self),
              let inputSizeBuffer = makeBuffer([inputSize]),
              let outputSizeBuffer = makeBuffer([outputSize]) else {
            throw ParagonError.bufferAllocationFailed
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ParagonError.commandEncodingFailed
        }

        encoder.setComputePipelineState(denseForwardPipeline!)
        encoder.setBuffer(weightsBuffer, offset: 0, index: 0)
        encoder.setBuffer(biasesBuffer, offset: 0, index: 1)
        encoder.setBuffer(inputBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBuffer(inputSizeBuffer, offset: 0, index: 4)
        encoder.setBuffer(outputSizeBuffer, offset: 0, index: 5)

        let gridSize = MTLSize(width: Int(outputSize), height: 1, depth: 1)
        let groupSize = MTLSize(width: min(THREADS_PER_GROUP, Int(outputSize)), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read results
        let resultPtr = outputBuffer.contents().bindMemory(to: Int32.self, capacity: Int(outputSize))
        let resultData = Array(UnsafeBufferPointer(start: resultPtr, count: Int(outputSize)))

        return Tensor(data: resultData, shape: [Int(outputSize)])
    }

    // MARK: - Backpropagation

    func computeOutputGradients(predictions: Tensor, targets: Tensor) throws -> Tensor {
        let size = UInt32(predictions.count)

        guard let predBuffer = makeBuffer(predictions.data),
              let targetBuffer = makeBuffer(targets.data),
              let gradBuffer = makeBuffer(count: Int(size), type: Int32.self),
              let sizeBuffer = makeBuffer([size]) else {
            throw ParagonError.bufferAllocationFailed
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ParagonError.commandEncodingFailed
        }

        encoder.setComputePipelineState(outputGradientsPipeline!)
        encoder.setBuffer(predBuffer, offset: 0, index: 0)
        encoder.setBuffer(targetBuffer, offset: 0, index: 1)
        encoder.setBuffer(gradBuffer, offset: 0, index: 2)
        encoder.setBuffer(sizeBuffer, offset: 0, index: 3)

        let gridSize = MTLSize(width: Int(size), height: 1, depth: 1)
        let groupSize = MTLSize(width: min(THREADS_PER_GROUP, Int(size)), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let resultPtr = gradBuffer.contents().bindMemory(to: Int32.self, capacity: Int(size))
        let resultData = Array(UnsafeBufferPointer(start: resultPtr, count: Int(size)))

        return Tensor(data: resultData, shape: predictions.shape)
    }

    func denseBackward(layer: DenseLayer, upstream: Tensor, input: Tensor)
        throws -> (weightGrads: Tensor, biasGrads: Tensor, inputGrads: Tensor) {

        let inputSize = UInt32(layer.weights.shape[1])
        let outputSize = UInt32(layer.weights.shape[0])

        // Allocate buffers
        guard let upstreamBuffer = makeBuffer(upstream.data),
              let layerInputBuffer = makeBuffer(input.data),
              let weightsBuffer = makeBuffer(layer.weights.data),
              let weightGradBuffer = makeBuffer(count: Int(inputSize * outputSize), type: Int32.self),
              let biasGradBuffer = makeBuffer(count: Int(outputSize), type: Int32.self),
              let inputGradBuffer = makeBuffer(count: Int(inputSize), type: Int32.self),
              let inputSizeBuffer = makeBuffer([inputSize]),
              let outputSizeBuffer = makeBuffer([outputSize]) else {
            throw ParagonError.bufferAllocationFailed
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw ParagonError.commandEncodingFailed
        }

        // Weight gradients (parallel over all weights)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(weightGradientsPipeline!)
            encoder.setBuffer(upstreamBuffer, offset: 0, index: 0)
            encoder.setBuffer(layerInputBuffer, offset: 0, index: 1)
            encoder.setBuffer(weightGradBuffer, offset: 0, index: 2)
            encoder.setBuffer(inputSizeBuffer, offset: 0, index: 3)
            encoder.setBuffer(outputSizeBuffer, offset: 0, index: 4)

            let gridSize = MTLSize(width: Int(outputSize), height: Int(inputSize), depth: 1)
            let groupSize = MTLSize(width: min(16, Int(outputSize)),
                                   height: min(16, Int(inputSize)), depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)
            encoder.endEncoding()
        }

        // Bias gradients
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(biasGradientsPipeline!)
            encoder.setBuffer(upstreamBuffer, offset: 0, index: 0)
            encoder.setBuffer(biasGradBuffer, offset: 0, index: 1)
            encoder.setBuffer(outputSizeBuffer, offset: 0, index: 2)

            let gridSize = MTLSize(width: Int(outputSize), height: 1, depth: 1)
            let groupSize = MTLSize(width: min(THREADS_PER_GROUP, Int(outputSize)), height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)
            encoder.endEncoding()
        }

        // Input gradients (for previous layer)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(inputGradientsPipeline!)
            encoder.setBuffer(upstreamBuffer, offset: 0, index: 0)
            encoder.setBuffer(weightsBuffer, offset: 0, index: 1)
            encoder.setBuffer(inputGradBuffer, offset: 0, index: 2)
            encoder.setBuffer(inputSizeBuffer, offset: 0, index: 3)
            encoder.setBuffer(outputSizeBuffer, offset: 0, index: 4)

            let gridSize = MTLSize(width: Int(inputSize), height: 1, depth: 1)
            let groupSize = MTLSize(width: min(THREADS_PER_GROUP, Int(inputSize)), height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)
            encoder.endEncoding()
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read results
        let weightGradPtr = weightGradBuffer.contents().bindMemory(to: Int32.self,
                                                                   capacity: Int(inputSize * outputSize))
        let biasGradPtr = biasGradBuffer.contents().bindMemory(to: Int32.self,
                                                               capacity: Int(outputSize))
        let inputGradPtr = inputGradBuffer.contents().bindMemory(to: Int32.self,
                                                                 capacity: Int(inputSize))

        let weightGrads = Tensor(
            data: Array(UnsafeBufferPointer(start: weightGradPtr, count: Int(inputSize * outputSize))),
            shape: layer.weights.shape
        )
        let biasGrads = Tensor(
            data: Array(UnsafeBufferPointer(start: biasGradPtr, count: Int(outputSize))),
            shape: layer.biases.shape
        )
        let inputGrads = Tensor(
            data: Array(UnsafeBufferPointer(start: inputGradPtr, count: Int(inputSize))),
            shape: input.shape
        )

        return (weightGrads, biasGrads, inputGrads)
    }

    // MARK: - Weight Update

    func sgdUpdate(layer: inout DenseLayer, weightGrads: Tensor, biasGrads: Tensor,
                   learningRate: Int32 = 100, momentum: Int32 = 900) throws {

        // Update weights
        try updateParameter(params: &layer.weights.data,
                           grads: weightGrads.data,
                           momentum: &layer.weightMomentum.data,
                           learningRate: learningRate,
                           momentumFactor: momentum)

        // Update biases
        try updateParameter(params: &layer.biases.data,
                           grads: biasGrads.data,
                           momentum: &layer.biasMomentum.data,
                           learningRate: learningRate,
                           momentumFactor: momentum)
    }

    private func updateParameter(params: inout [Int32], grads: [Int32],
                                 momentum: inout [Int32],
                                 learningRate: Int32, momentumFactor: Int32) throws {
        let size = UInt32(params.count)

        guard let paramBuffer = makeBuffer(params),
              let gradBuffer = makeBuffer(grads),
              let momentumBuffer = makeBuffer(momentum),
              let lrBuffer = makeBuffer([learningRate]),
              let mfBuffer = makeBuffer([momentumFactor]),
              let sizeBuffer = makeBuffer([size]) else {
            throw ParagonError.bufferAllocationFailed
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ParagonError.commandEncodingFailed
        }

        encoder.setComputePipelineState(sgdUpdatePipeline!)
        encoder.setBuffer(paramBuffer, offset: 0, index: 0)
        encoder.setBuffer(gradBuffer, offset: 0, index: 1)
        encoder.setBuffer(momentumBuffer, offset: 0, index: 2)
        encoder.setBuffer(lrBuffer, offset: 0, index: 3)
        encoder.setBuffer(mfBuffer, offset: 0, index: 4)
        encoder.setBuffer(sizeBuffer, offset: 0, index: 5)

        let gridSize = MTLSize(width: Int(size), height: 1, depth: 1)
        let groupSize = MTLSize(width: min(THREADS_PER_GROUP, Int(size)), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read updated values
        let paramPtr = paramBuffer.contents().bindMemory(to: Int32.self, capacity: Int(size))
        let momentumPtr = momentumBuffer.contents().bindMemory(to: Int32.self, capacity: Int(size))

        params = Array(UnsafeBufferPointer(start: paramPtr, count: Int(size)))
        momentum = Array(UnsafeBufferPointer(start: momentumPtr, count: Int(size)))
    }

    // MARK: - Loss Computation

    func mseLoss(predictions: Tensor, targets: Tensor) throws -> Int32 {
        let size = UInt32(predictions.count)
        let numGroups = (Int(size) + THREADS_PER_GROUP - 1) / THREADS_PER_GROUP

        guard let predBuffer = makeBuffer(predictions.data),
              let targetBuffer = makeBuffer(targets.data),
              let partialBuffer = makeBuffer(count: numGroups, type: Int32.self),
              let sizeBuffer = makeBuffer([size]) else {
            throw ParagonError.bufferAllocationFailed
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ParagonError.commandEncodingFailed
        }

        encoder.setComputePipelineState(mseLossPipeline!)
        encoder.setBuffer(predBuffer, offset: 0, index: 0)
        encoder.setBuffer(targetBuffer, offset: 0, index: 1)
        encoder.setBuffer(partialBuffer, offset: 0, index: 2)
        encoder.setBuffer(sizeBuffer, offset: 0, index: 3)

        let gridSize = MTLSize(width: Int(size), height: 1, depth: 1)
        let groupSize = MTLSize(width: THREADS_PER_GROUP, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Sum partial losses on CPU (small number of groups)
        let partialPtr = partialBuffer.contents().bindMemory(to: Int32.self, capacity: numGroups)
        var totalLoss: Int32 = 0
        for i in 0..<numGroups {
            totalLoss += partialPtr[i]
        }

        return totalLoss / Int32(predictions.count)
    }

    // MARK: - Device Info

    var deviceInfo: String {
        """
        Device: \(device.name)
        Max Threads Per Threadgroup: \(device.maxThreadsPerThreadgroup)
        Unified Memory: \(device.hasUnifiedMemory)
        Recommended Max Working Set Size: \(device.recommendedMaxWorkingSetSize / 1_000_000) MB
        """
    }
}

// MARK: - Neural Network

class NeuralNetwork {
    var layers: [DenseLayer]
    let metal: ParagonMetal

    // Cache for backprop
    var activations: [Tensor] = []

    init(layerSizes: [Int], metal: ParagonMetal) {
        self.metal = metal
        self.layers = []

        for i in 0..<(layerSizes.count - 1) {
            layers.append(DenseLayer(inputSize: layerSizes[i],
                                    outputSize: layerSizes[i + 1],
                                    seed: UInt64(42 + i)))
        }
    }

    func forward(_ input: Tensor) throws -> Tensor {
        activations = [input]
        var current = input

        for layer in layers {
            current = try metal.denseForward(layer: layer, input: current)
            activations.append(current)
        }

        return current
    }

    func backward(target: Tensor, learningRate: Int32 = 100) throws {
        guard let predictions = activations.last else { return }

        // Compute output gradients
        var upstream = try metal.computeOutputGradients(predictions: predictions, targets: target)

        // Backpropagate through layers (reverse order)
        for i in (0..<layers.count).reversed() {
            let input = activations[i]
            let (weightGrads, biasGrads, inputGrads) = try metal.denseBackward(
                layer: layers[i],
                upstream: upstream,
                input: input
            )

            // Update weights
            try metal.sgdUpdate(layer: &layers[i],
                               weightGrads: weightGrads,
                               biasGrads: biasGrads,
                               learningRate: learningRate)

            upstream = inputGrads
        }
    }

    func train(inputs: [Tensor], targets: [Tensor], epochs: Int, learningRate: Int32 = 100) throws {
        for epoch in 0..<epochs {
            var totalLoss: Int32 = 0

            for (input, target) in zip(inputs, targets) {
                let output = try forward(input)
                let loss = try metal.mseLoss(predictions: output, targets: target)
                totalLoss += loss

                try backward(target: target, learningRate: learningRate)
            }

            if epoch % 100 == 0 {
                print("Epoch \(epoch): Loss = \(Double(totalLoss) / Double(SCALE_FACTOR) / Double(inputs.count))")
            }
        }
    }
}

// MARK: - Example Usage

func runXORExample() {
    do {
        let metal = try ParagonMetal()
        print(metal.deviceInfo)
        print()

        // Create XOR dataset (scaled by 1000)
        let inputs = [
            Tensor(data: [0, 0], shape: [2]),
            Tensor(data: [0, 1000], shape: [2]),
            Tensor(data: [1000, 0], shape: [2]),
            Tensor(data: [1000, 1000], shape: [2])
        ]
        let targets = [
            Tensor(data: [0], shape: [1]),
            Tensor(data: [1000], shape: [1]),
            Tensor(data: [1000], shape: [1]),
            Tensor(data: [0], shape: [1])
        ]

        // Create network: 2 -> 8 -> 1
        let network = NeuralNetwork(layerSizes: [2, 8, 1], metal: metal)

        print("Training XOR network on Apple Silicon GPU...")
        let start = CFAbsoluteTimeGetCurrent()

        try network.train(inputs: inputs, targets: targets, epochs: 1000, learningRate: 50)

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        print("\nTraining completed in \(String(format: "%.3f", elapsed)) seconds")

        // Test
        print("\nResults:")
        for (input, target) in zip(inputs, targets) {
            let output = try network.forward(input)
            let inputStr = input.data.map { $0 == 1000 ? "1" : "0" }.joined(separator: ", ")
            let expected = target.data[0] == 1000 ? "1" : "0"
            let actual = Double(output.data[0]) / Double(SCALE_FACTOR)
            print("  Input: [\(inputStr)] -> Expected: \(expected), Got: \(String(format: "%.3f", actual))")
        }

    } catch {
        print("Error: \(error)")
    }
}

// Run if executed directly
// runXORExample()
