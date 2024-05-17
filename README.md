# Paragon

Provably correct, high-performance model training using Bend's massively parallel capabilities.

## Overview

Paragon leverages the powerful capabilities of the Bend programming language, built on the Higher-order Virtual Machine 2 (HVM2), to enable efficient, decentralized, and parallel training of AI models. 

## Key Features

- **Provably Correct Training**: Ensuring the accuracy and correctness of AI model training through formal verification methods.
- **Decentralized Architecture**: Training models across a distributed network of machines, enhancing robustness and fault tolerance.
- **Massively Parallel Execution**: Utilizing the Bend programming language to achieve near-linear speedup with core count, running efficiently on GPU hardware.
- **Ease of Use**: High-level language features of Bend, inspired by Python and Haskell, allow for expressive and concise code without the need for explicit parallel annotations.

## Why Bend?

Bend is designed to combine the ease of high-level programming with the performance of low-level parallel execution. Key advantages include:

- **Expressiveness**: Enjoy the syntactic and functional richness of languages like Python and Haskell.
- **Performance**: Achieve fast, massively parallel computation on GPUs.
- **Simplicity**: Write parallel programs without dealing with threads, locks, or other concurrency primitives.
- **Scalability**: Scale your AI model training linearly with the number of GPU cores.

## Getting Started

To get started with this project, follow these steps:

### Prerequisites

- **Bend Language**: Install the Bend programming language from [the official repository](https://github.com/HigherOrderCO/Bend).
- **HVM2 Runtime**: Ensure you have the HVM2 runtime environment set up on your system.

### Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/lulzx/paragon.git
cd paragon
```

### Running the Examples

We have provided example scripts to demonstrate the capabilities of decentralized parallel AI model training using Bend. To run an example, execute:

```bash
bend run examples/training_example.bend
```

### Building from Source

If you want to build the project from source, use the provided build script:

```bash
./build.sh
```

This will compile the low-level IR language to C and CUDA, and generate the necessary binaries for running on GPU hardware.

## Documentation

For detailed documentation on the Bend language and HVM2 runtime, refer to:

- [Bend Language Documentation](https://github.com/HigherOrderCO/Bend/tree/main/docs)
- [HVM2 Runtime Paper](https://github.com/HigherOrderCO/HVM/blob/main/paper%2FPAPER.pdf)

## Contributing

We welcome contributions from the community. If you have ideas, bug fixes, or enhancements, please open an issue or submit a pull request.

### Reporting Issues

If you encounter any issues or have questions, feel free to open an issue on our [GitHub Issues page](https://github.com/lulzx/paragon/issues).

### Code of Conduct

Please adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) in all your interactions with the project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to the developers of Bend and HVM2 for creating the underlying technologies that power this project.

This project started [here X](https://twitter.com/Wooltard/status/1791558096420032959)

---

We hope this project empowers you to harness the full potential of decentralized, parallel AI model training. Happy coding!
