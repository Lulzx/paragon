#!/usr/bin/env swift
// Paragon Metal Benchmark Suite
// Measures GPU performance on Apple Silicon

import Metal
import Foundation

// MARK: - Constants

let SCALE_FACTOR: Int32 = 1000
let THREADS_PER_GROUP = 256

// MARK: - Benchmark Results

struct BenchmarkResult {
    let name: String
    let operations: Int
    let timeSeconds: Double
    let mips: Double
}

// MARK: - Metal Benchmark Runner

class MetalBenchmark {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary

    var results: [BenchmarkResult] = []

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "MetalBenchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: "Metal not available"])
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw NSError(domain: "MetalBenchmark", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create command queue"])
        }
        self.commandQueue = queue

        // Compile shaders from source
        let shaderPath = "metal/shaders.metal"
        let source = try String(contentsOfFile: shaderPath, encoding: .utf8)
        self.library = try device.makeLibrary(source: source, options: nil)
    }

    func makeBuffer<T>(_ data: [T]) -> MTLBuffer? {
        device.makeBuffer(bytes: data, length: MemoryLayout<T>.stride * data.count, options: .storageModeShared)
    }

    func makeBuffer<T>(count: Int, type: T.Type) -> MTLBuffer? {
        device.makeBuffer(length: MemoryLayout<T>.stride * count, options: .storageModeShared)
    }

    // MARK: - Benchmark: Tensor Add

    func benchmarkTensorAdd(size: Int) throws -> BenchmarkResult {
        guard let function = library.makeFunction(name: "tensor_add"),
              let pipeline = try? device.makeComputePipelineState(function: function) else {
            throw NSError(domain: "MetalBenchmark", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create pipeline"])
        }

        var dataA = [Int32](repeating: 0, count: size)
        var dataB = [Int32](repeating: 0, count: size)
        for i in 0..<size {
            dataA[i] = Int32(i % 1000)
            dataB[i] = Int32((i * 7) % 1000)
        }

        guard let bufferA = makeBuffer(dataA),
              let bufferB = makeBuffer(dataB),
              let bufferResult = makeBuffer(count: size, type: Int32.self),
              let sizeBuffer = makeBuffer([UInt32(size)]) else {
            throw NSError(domain: "MetalBenchmark", code: 4, userInfo: [NSLocalizedDescriptionKey: "Buffer allocation failed"])
        }

        // Warm up
        for _ in 0..<5 {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(bufferA, offset: 0, index: 0)
            encoder.setBuffer(bufferB, offset: 0, index: 1)
            encoder.setBuffer(bufferResult, offset: 0, index: 2)
            encoder.setBuffer(sizeBuffer, offset: 0, index: 3)

            let gridSize = MTLSize(width: size, height: 1, depth: 1)
            let groupSize = MTLSize(width: min(THREADS_PER_GROUP, size), height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)
            encoder.endEncoding()

            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        // Benchmark
        let iterations = 100
        let start = CFAbsoluteTimeGetCurrent()

        for _ in 0..<iterations {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(bufferA, offset: 0, index: 0)
            encoder.setBuffer(bufferB, offset: 0, index: 1)
            encoder.setBuffer(bufferResult, offset: 0, index: 2)
            encoder.setBuffer(sizeBuffer, offset: 0, index: 3)

            let gridSize = MTLSize(width: size, height: 1, depth: 1)
            let groupSize = MTLSize(width: min(THREADS_PER_GROUP, size), height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)
            encoder.endEncoding()

            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgTime = elapsed / Double(iterations)
        let operations = size
        let mips = Double(operations) / avgTime / 1_000_000

        return BenchmarkResult(
            name: "Tensor Add (\(size/1024)K)",
            operations: operations,
            timeSeconds: avgTime,
            mips: mips
        )
    }

    // MARK: - Benchmark: ReLU

    func benchmarkReLU(size: Int) throws -> BenchmarkResult {
        guard let function = library.makeFunction(name: "tensor_relu"),
              let pipeline = try? device.makeComputePipelineState(function: function) else {
            throw NSError(domain: "MetalBenchmark", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create pipeline"])
        }

        var input = [Int32](repeating: 0, count: size)
        for i in 0..<size {
            input[i] = Int32(i % 2000) - 1000
        }

        guard let inputBuffer = makeBuffer(input),
              let outputBuffer = makeBuffer(count: size, type: Int32.self),
              let sizeBuffer = makeBuffer([UInt32(size)]) else {
            throw NSError(domain: "MetalBenchmark", code: 4, userInfo: [NSLocalizedDescriptionKey: "Buffer allocation failed"])
        }

        // Warm up
        for _ in 0..<5 {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBuffer(sizeBuffer, offset: 0, index: 2)

            let gridSize = MTLSize(width: size, height: 1, depth: 1)
            let groupSize = MTLSize(width: min(THREADS_PER_GROUP, size), height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)
            encoder.endEncoding()

            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        // Benchmark
        let iterations = 100
        let start = CFAbsoluteTimeGetCurrent()

        for _ in 0..<iterations {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBuffer(sizeBuffer, offset: 0, index: 2)

            let gridSize = MTLSize(width: size, height: 1, depth: 1)
            let groupSize = MTLSize(width: min(THREADS_PER_GROUP, size), height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)
            encoder.endEncoding()

            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgTime = elapsed / Double(iterations)
        let operations = size
        let mips = Double(operations) / avgTime / 1_000_000

        return BenchmarkResult(
            name: "ReLU (\(size/1024)K)",
            operations: operations,
            timeSeconds: avgTime,
            mips: mips
        )
    }

    // MARK: - Benchmark: Dense Forward

    func benchmarkDenseForward(inputSize: Int, outputSize: Int) throws -> BenchmarkResult {
        guard let function = library.makeFunction(name: "dense_forward"),
              let pipeline = try? device.makeComputePipelineState(function: function) else {
            throw NSError(domain: "MetalBenchmark", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create pipeline"])
        }

        var weights = [Int32](repeating: 0, count: inputSize * outputSize)
        var biases = [Int32](repeating: 0, count: outputSize)
        var input = [Int32](repeating: 0, count: inputSize)

        for i in 0..<weights.count {
            weights[i] = Int32((i % 200) - 100)
        }
        for i in 0..<biases.count {
            biases[i] = Int32(i % 50)
        }
        for i in 0..<input.count {
            input[i] = Int32((i % 100) * 10)
        }

        guard let weightsBuffer = makeBuffer(weights),
              let biasesBuffer = makeBuffer(biases),
              let inputBuffer = makeBuffer(input),
              let outputBuffer = makeBuffer(count: outputSize, type: Int32.self),
              let inputSizeBuffer = makeBuffer([UInt32(inputSize)]),
              let outputSizeBuffer = makeBuffer([UInt32(outputSize)]) else {
            throw NSError(domain: "MetalBenchmark", code: 4, userInfo: [NSLocalizedDescriptionKey: "Buffer allocation failed"])
        }

        // Warm up
        for _ in 0..<5 {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(weightsBuffer, offset: 0, index: 0)
            encoder.setBuffer(biasesBuffer, offset: 0, index: 1)
            encoder.setBuffer(inputBuffer, offset: 0, index: 2)
            encoder.setBuffer(outputBuffer, offset: 0, index: 3)
            encoder.setBuffer(inputSizeBuffer, offset: 0, index: 4)
            encoder.setBuffer(outputSizeBuffer, offset: 0, index: 5)

            let gridSize = MTLSize(width: outputSize, height: 1, depth: 1)
            let groupSize = MTLSize(width: min(THREADS_PER_GROUP, outputSize), height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)
            encoder.endEncoding()

            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        // Benchmark
        let iterations = 100
        let start = CFAbsoluteTimeGetCurrent()

        for _ in 0..<iterations {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(weightsBuffer, offset: 0, index: 0)
            encoder.setBuffer(biasesBuffer, offset: 0, index: 1)
            encoder.setBuffer(inputBuffer, offset: 0, index: 2)
            encoder.setBuffer(outputBuffer, offset: 0, index: 3)
            encoder.setBuffer(inputSizeBuffer, offset: 0, index: 4)
            encoder.setBuffer(outputSizeBuffer, offset: 0, index: 5)

            let gridSize = MTLSize(width: outputSize, height: 1, depth: 1)
            let groupSize = MTLSize(width: min(THREADS_PER_GROUP, outputSize), height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)
            encoder.endEncoding()

            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgTime = elapsed / Double(iterations)
        let operations = inputSize * outputSize + outputSize
        let mips = Double(operations) / avgTime / 1_000_000

        return BenchmarkResult(
            name: "Dense (\(inputSize)->\(outputSize))",
            operations: operations,
            timeSeconds: avgTime,
            mips: mips
        )
    }

    // MARK: - Run All Benchmarks

    func runAll() {
        print("================================================================================")
        print("Paragon Metal Benchmark Suite")
        print("================================================================================")
        print("")
        print("Device: \(device.name)")
        print("Unified Memory: \(device.hasUnifiedMemory)")
        print("Max Threads Per Threadgroup: \(device.maxThreadsPerThreadgroup)")
        print("")
        print("--------------------------------------------------------------------------------")

        // Tensor Add
        do {
            let r1 = try benchmarkTensorAdd(size: 1024)
            results.append(r1)
            print("Tensor Add (1K): \(Int(r1.mips)) MIPS")

            let r2 = try benchmarkTensorAdd(size: 65536)
            results.append(r2)
            print("Tensor Add (64K): \(Int(r2.mips)) MIPS")

            let r3 = try benchmarkTensorAdd(size: 1048576)
            results.append(r3)
            print("Tensor Add (1M): \(Int(r3.mips)) MIPS")
        } catch {
            print("Tensor Add benchmark failed: \(error)")
        }

        // ReLU
        do {
            let r1 = try benchmarkReLU(size: 1024)
            results.append(r1)
            print("ReLU (1K): \(Int(r1.mips)) MIPS")

            let r2 = try benchmarkReLU(size: 65536)
            results.append(r2)
            print("ReLU (64K): \(Int(r2.mips)) MIPS")

            let r3 = try benchmarkReLU(size: 1048576)
            results.append(r3)
            print("ReLU (1M): \(Int(r3.mips)) MIPS")
        } catch {
            print("ReLU benchmark failed: \(error)")
        }

        // Dense Forward
        do {
            let r1 = try benchmarkDenseForward(inputSize: 64, outputSize: 64)
            results.append(r1)
            print("Dense (64->64): \(Int(r1.mips)) MIPS")

            let r2 = try benchmarkDenseForward(inputSize: 256, outputSize: 256)
            results.append(r2)
            print("Dense (256->256): \(Int(r2.mips)) MIPS")

            let r3 = try benchmarkDenseForward(inputSize: 1024, outputSize: 512)
            results.append(r3)
            print("Dense (1024->512): \(Int(r3.mips)) MIPS")
        } catch {
            print("Dense Forward benchmark failed: \(error)")
        }

        print("--------------------------------------------------------------------------------")
        print("")

        // Summary
        if !results.isEmpty {
            let avgMips = results.reduce(0.0) { $0 + $1.mips } / Double(results.count)
            let maxResult = results.max(by: { $0.mips < $1.mips })!

            print("Summary:")
            print("  Total benchmarks: \(results.count)")
            print("  Average MIPS: \(Int(avgMips))")
            print("  Peak MIPS: \(Int(maxResult.mips)) (\(maxResult.name))")
            print("")

            // Markdown output
            print("## Markdown for BENCHMARKS.md")
            print("")
            print("| Benchmark | Operations | Time (s) | Performance (MIPS) |")
            print("|-----------|------------|----------|-------------------|")
            for r in results {
                let timeStr = String(format: "%.6f", r.timeSeconds)
                print("| \(r.name) | \(r.operations) | \(timeStr) | **\(Int(r.mips))** |")
            }
        }
    }
}

// MARK: - Main

do {
    let benchmark = try MetalBenchmark()
    benchmark.runAll()
} catch {
    print("Failed to initialize Metal: \(error)")
    exit(1)
}
