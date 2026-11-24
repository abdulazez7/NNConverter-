# RVForge

Convert supported PyTorch neural networks into portable C and RISC-V test executables.

RVForge captures your model, generates C, and verifies the outputs. Choose FP32, INT8 or binary weight storage, with support for dense networks, convolutions, embeddings and attention operations.

## Use

Install the package and provide a Python model file containing a `build()` function that returns the model and example inputs.

```sh
python -m pip install -e .
python -m rvforge convert my_model.py --precision int8 --out build/my-model
```

The output includes C source, a model header, verification fixtures and a manifest. Add `--zig /path/to/zig --riscv` to compile and run a test executable in the included RISC-V emulator. Compressed modes retain floating-point arithmetic.

See the [usage guide](docs/USAGE.md) for model templates, requirements and target integration.

## Documentation

[Usage](docs/USAGE.md) · [Supported operations](docs/OPERATORS.md) · [Architecture](docs/ARCHITECTURE.md) · [Validation](docs/VALIDATION.md)

## License

MIT. See [LICENSE](LICENSE).
