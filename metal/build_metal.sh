#!/bin/bash
# Build script for Paragon Metal backend on Apple Silicon
# Compiles Metal shaders and Swift runtime

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"

echo "=== Paragon Metal Build Script ==="
echo "Building for Apple Silicon GPU..."
echo ""

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "Warning: Not running on Apple Silicon (arm64)"
    echo "Metal GPU acceleration requires Apple Silicon Mac"
fi

# Check for Metal compiler
if ! command -v xcrun &> /dev/null; then
    echo "Error: Xcode command line tools not found"
    echo "Install with: xcode-select --install"
    exit 1
fi

# Create build directory
mkdir -p "${BUILD_DIR}"

echo "Step 1: Compiling Metal shaders..."
xcrun -sdk macosx metal -c "${SCRIPT_DIR}/shaders.metal" -o "${BUILD_DIR}/shaders.air"

echo "Step 2: Creating Metal library..."
xcrun -sdk macosx metallib "${BUILD_DIR}/shaders.air" -o "${BUILD_DIR}/shaders.metallib"

echo "Step 3: Compiling Swift runtime..."
swiftc -O \
    -sdk $(xcrun --sdk macosx --show-sdk-path) \
    -target arm64-apple-macosx13.0 \
    "${SCRIPT_DIR}/ParagonMetal.swift" \
    -emit-library \
    -module-name ParagonMetal \
    -o "${BUILD_DIR}/libParagonMetal.dylib"

# Create a standalone executable for testing
echo "Step 4: Building test executable..."
cat > "${BUILD_DIR}/test_metal.swift" << 'EOF'
import Foundation

// Simple test to verify Metal runtime works
print("Paragon Metal Test")
print("==================")

// Check for Metal support
if let device = MTLCreateSystemDefaultDevice() {
    print("✓ Metal device found: \(device.name)")
    print("  - Unified Memory: \(device.hasUnifiedMemory)")
    print("  - Max Threads/Group: \(device.maxThreadsPerThreadgroup)")
    print("  - Recommended Working Set: \(device.recommendedMaxWorkingSetSize / 1_000_000) MB")
} else {
    print("✗ No Metal device found")
    exit(1)
}

print("\n✓ Metal backend ready for use")
EOF

swiftc -O \
    -sdk $(xcrun --sdk macosx --show-sdk-path) \
    -target arm64-apple-macosx13.0 \
    "${BUILD_DIR}/test_metal.swift" \
    -framework Metal \
    -framework MetalPerformanceShaders \
    -o "${BUILD_DIR}/test_metal"

echo ""
echo "=== Build Complete ==="
echo ""
echo "Generated files:"
echo "  ${BUILD_DIR}/shaders.metallib    - Compiled Metal shaders"
echo "  ${BUILD_DIR}/libParagonMetal.dylib - Swift runtime library"
echo "  ${BUILD_DIR}/test_metal          - Test executable"
echo ""
echo "To test Metal support:"
echo "  ${BUILD_DIR}/test_metal"
echo ""
echo "To use in your project:"
echo "  1. Link against libParagonMetal.dylib"
echo "  2. Include shaders.metallib in your bundle"
echo "  3. Import ParagonMetal module"
