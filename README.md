# Paragon

Provably correct, high-performance model training using Bend's massively parallel capabilities.

## Overview

Paragon leverages the powerful capabilities of the Bend programming language, built on the Higher-order Virtual Machine 3 (HVM3), to enable efficient, decentralized, and parallel training of AI models.

HVM3 is an efficient implementation of the Interaction Calculus (IC), a computational paradigm that serves as an alternative foundation to Lambda Calculus. It delivers optimal beta-reduction with potential exponential speedups for certain expressions.

## Key Features

- **Provably Correct Training**: Ensuring the accuracy and correctness of AI model training through formal verification methods.
- **Decentralized Architecture**: Training models across a distributed network of machines, enhancing robustness and fault tolerance.
- **Massively Parallel Execution**: Utilizing the Bend programming language to achieve near-linear speedup with core count, running efficiently on GPU hardware.
- **Ease of Use**: High-level language features of Bend, inspired by Python and Haskell, allow for expressive and concise code without the need for explicit parallel annotations.

## Why HVM3?

HVM3 introduces several unique computational primitives that set it apart from traditional functional runtimes:

- **Linear/Affine Lambdas**: Functions are resource-aware, enabling efficient memory management.
- **First-Class Duplications**: Terms can be copied into multiple locations natively.
- **First-Class Superpositions**: Multiple terms can occupy a single location simultaneously.
- **Scope-Free Lambdas**: Permits global substitutions without traditional scope boundaries.

These features enable native representation of continuations, linear higher-order abstract syntax (HOAS) interpreters, and mutable references. The fully affine design supports efficient garbage collection and simplified parallelism.

## Why Bend?

Bend is designed to combine the ease of high-level programming with the performance of low-level parallel execution. Key advantages include:

- **Expressiveness**: Enjoy the syntactic and functional richness of languages like Python and Haskell.
- **Performance**: Achieve fast, massively parallel computation on GPUs.
- **Simplicity**: Write parallel programs without dealing with threads, locks, or other concurrency primitives.
- **Scalability**: Scale your AI model training linearly with the number of GPU cores.

## Getting Started

To get started with this project, follow these steps:

### Prerequisites

- **Cabal**: Install Haskell's build system from [haskell.org](https://www.haskell.org/cabal/).
- **Bend Language**: Install the Bend programming language from [the official repository](https://github.com/HigherOrderCO/Bend).
- **HVM3 Runtime**: Install the HVM3 runtime environment:

```bash
git clone https://github.com/HigherOrderCO/HVM3.git
cd HVM3
cabal install
```

### Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/lulzx/paragon.git
cd paragon
```

### Running the Examples

We have provided example scripts to demonstrate the capabilities of decentralized parallel AI model training. To run an example:

```bash
# Interpreted mode (slower, good for development)
hvm run examples/training_example.hvml

# Compiled mode (faster, recommended for production)
hvm run examples/training_example.hvml -c
```

The compiled flag (`-c`) generates a standalone C file that can be independently compiled and executed for maximum performance.

### Building from Source

If you want to build the project from source, use the provided build script:

```bash
./build.sh
```

This will compile the low-level IR language to C and CUDA, and generate the necessary binaries for running on GPU hardware.

## Documentation

For detailed documentation on the Bend language and HVM3 runtime, refer to:

- [Bend Language Documentation](https://github.com/HigherOrderCO/Bend/tree/main/docs)
- [HVM3 Repository](https://github.com/HigherOrderCO/HVM3)
- [Interaction Calculus Specification (IC.md)](https://github.com/HigherOrderCO/HVM3/blob/main/IC.md)
- [HVM Language Specification (HVM.md)](https://github.com/HigherOrderCO/HVM3/blob/main/HVM.md)

## Contributing

We welcome contributions from the community. If you have ideas, bug fixes, or enhancements, please open an issue or submit a pull request.

### Reporting Issues

If you encounter any issues or have questions, feel free to open an issue on our [GitHub Issues page](https://github.com/lulzx/paragon/issues).

### Code of Conduct

Please adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) in all your interactions with the project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to the developers of Bend and HVM3 for creating the underlying technologies that power this project.

This project started [here X](https://twitter.com/Wooltard/status/1791558096420032959)

---

We hope this project empowers you to harness the full potential of decentralized, parallel AI model training. Happy coding!
