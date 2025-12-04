#!/bin/bash
# Paragon Build Script
# Compiles Bend source files to C/CUDA for high-performance execution

set -e

echo "==================================="
echo "  Paragon Build System"
echo "==================================="
echo ""

# Configuration
SRC_DIR="src"
EXAMPLES_DIR="examples"
BUILD_DIR="build"
OUTPUT_DIR="bin"

# Check for required tools
check_dependencies() {
    echo "[1/5] Checking dependencies..."

    if ! command -v bend &> /dev/null; then
        echo "Error: Bend compiler not found."
        echo "Please install Bend from: https://github.com/HigherOrderCO/Bend"
        exit 1
    fi

    if ! command -v hvm &> /dev/null; then
        echo "Warning: HVM3 runtime not found."
        echo "Install from: https://github.com/HigherOrderCO/HVM3"
        echo "Continuing with Bend only..."
    fi

    if ! command -v gcc &> /dev/null && ! command -v clang &> /dev/null; then
        echo "Warning: No C compiler found. Compiled mode may not work."
    fi

    echo "  Dependencies OK"
    echo ""
}

# Create build directories
setup_directories() {
    echo "[2/5] Setting up build directories..."
    mkdir -p "$BUILD_DIR"
    mkdir -p "$OUTPUT_DIR"
    echo "  Created: $BUILD_DIR/, $OUTPUT_DIR/"
    echo ""
}

# Compile source library files
compile_sources() {
    echo "[3/5] Compiling source library..."

    if [ -d "$SRC_DIR" ]; then
        for file in "$SRC_DIR"/*.bend; do
            if [ -f "$file" ]; then
                filename=$(basename "$file" .bend)
                echo "  Compiling: $filename.bend"

                # Generate C code using Bend
                if command -v bend &> /dev/null; then
                    bend gen-c "$file" > "$BUILD_DIR/${filename}.c" 2>/dev/null || true
                fi
            fi
        done
    else
        echo "  No source files found in $SRC_DIR/"
    fi
    echo ""
}

# Compile examples
compile_examples() {
    echo "[4/5] Compiling examples..."

    if [ -d "$EXAMPLES_DIR" ]; then
        for file in "$EXAMPLES_DIR"/*.bend; do
            if [ -f "$file" ]; then
                filename=$(basename "$file" .bend)
                echo "  Compiling: $filename.bend"

                # Generate C code
                if command -v bend &> /dev/null; then
                    bend gen-c "$file" > "$BUILD_DIR/${filename}.c" 2>/dev/null || true
                fi

                # Compile C to binary if compiler available
                if command -v gcc &> /dev/null; then
                    if [ -f "$BUILD_DIR/${filename}.c" ]; then
                        gcc -O3 -o "$OUTPUT_DIR/$filename" "$BUILD_DIR/${filename}.c" 2>/dev/null || true
                    fi
                elif command -v clang &> /dev/null; then
                    if [ -f "$BUILD_DIR/${filename}.c" ]; then
                        clang -O3 -o "$OUTPUT_DIR/$filename" "$BUILD_DIR/${filename}.c" 2>/dev/null || true
                    fi
                fi
            fi
        done
    else
        echo "  No example files found in $EXAMPLES_DIR/"
    fi
    echo ""
}

# Build CUDA targets (if nvcc available)
compile_cuda() {
    echo "[5/5] Checking CUDA support..."

    if command -v nvcc &> /dev/null; then
        echo "  CUDA compiler found. Building GPU targets..."

        for file in "$BUILD_DIR"/*.c; do
            if [ -f "$file" ]; then
                filename=$(basename "$file" .c)
                # Generate CUDA version
                if command -v bend &> /dev/null; then
                    bend gen-cu "${EXAMPLES_DIR}/${filename}.bend" > "$BUILD_DIR/${filename}.cu" 2>/dev/null || true
                fi

                if [ -f "$BUILD_DIR/${filename}.cu" ]; then
                    nvcc -O3 -o "$OUTPUT_DIR/${filename}_cuda" "$BUILD_DIR/${filename}.cu" 2>/dev/null || true
                    echo "  Built: ${filename}_cuda"
                fi
            fi
        done
    else
        echo "  CUDA not available. Skipping GPU builds."
        echo "  Install CUDA toolkit for GPU acceleration."
    fi
    echo ""
}

# Print summary
print_summary() {
    echo "==================================="
    echo "  Build Complete!"
    echo "==================================="
    echo ""
    echo "Generated files:"

    if [ -d "$BUILD_DIR" ]; then
        echo "  C sources:  $(ls -1 $BUILD_DIR/*.c 2>/dev/null | wc -l | tr -d ' ') files in $BUILD_DIR/"
    fi

    if [ -d "$OUTPUT_DIR" ]; then
        echo "  Binaries:   $(ls -1 $OUTPUT_DIR/* 2>/dev/null | wc -l | tr -d ' ') files in $OUTPUT_DIR/"
    fi

    echo ""
    echo "Run examples:"
    echo "  Interpreted:  bend run examples/training_example.bend"
    echo "  Parallel C:   bend run-c examples/training_example.bend"
    echo "  Parallel GPU: bend run-cu examples/training_example.bend"
    echo ""
    echo "Or with HVM3:"
    echo "  Interpreted:  hvm run examples/training_example.bend"
    echo "  Compiled:     hvm run examples/training_example.bend -c"
    echo ""
}

# Clean build artifacts
clean() {
    echo "Cleaning build artifacts..."
    rm -rf "$BUILD_DIR"
    rm -rf "$OUTPUT_DIR"
    echo "Done."
}

# Main build process
main() {
    case "${1:-}" in
        clean)
            clean
            ;;
        *)
            check_dependencies
            setup_directories
            compile_sources
            compile_examples
            compile_cuda
            print_summary
            ;;
    esac
}

main "$@"
