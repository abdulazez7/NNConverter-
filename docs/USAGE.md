# RVForge usage guide

Convert a neural network written in Python into portable C and a runnable RISC-V test executable. Supply a PyTorch module and example inputs; RVForge captures its operations, compresses eligible weights, generates C, and checks its outputs before executing the firmware.

The compiler works with combinations of supported operations. It does not recognize or substitute a list of predefined architectures. Examples cover dense networks, 1D and 2D CNNs, grouped convolutions, residual paths, attention, embeddings, and multiple inputs and outputs.

**Status:** experimental 0.1.0. The supplied target is RV32IMF with Zicsr on the QEMU `virt` memory map. Physical-board deployment needs that board's startup code, linker script, I/O and memory budget. The included ELF is a self-test application with fixed fixtures, not an ESP32 firmware image or a complete LLM chat runtime.

## Installation and conversion

Install Python 3.10 or later, then run from this directory:

```sh
python -m pip install -e .
python -m rvforge convert examples/mlp.py --precision int8 --out build/my-model
```

This creates `model.c`, `model.h`, verification fixtures, and a manifest describing shapes, memory use and compression error. You do not need to prepare an intermediate model file.

For compilation and execution, install **Zig 0.16** and **Node.js 20 or later**:

```sh
python -m rvforge convert examples/sequence_transformer.py --precision int8 --out build/sequence --zig /path/to/zig --riscv
```

On Windows, use the full path to `zig.exe`. The command compiles and runs a host self-test, cross-compiles `model-rv32.elf`, and runs that executable in the included functional RISC-V emulator. Each output element must match the independent reference within the documented tolerance.

To run an existing executable without Python or a compiler:

```sh
node rvforge/target/emulate.js prebuilt/transformer-int8/model-rv32.elf
```

For independent execution with QEMU installed:

```sh
qemu-system-riscv32 -M virt -cpu rv32 -m 16M -bios none -device loader,file=prebuilt/transformer-int8/model-rv32.elf,cpu-num=0 -nographic -monitor none
```

The release was verified using the included functional emulator. The QEMU command is supplied for independent reproduction; it was not executed in this release environment. Emulator instruction counts and host elapsed time are not hardware cycle or latency measurements.

## Use your own Python model

Create `my_model.py`:

```python
import torch
from torch import nn

def build():
    model = nn.Sequential(nn.Linear(12, 24), nn.ReLU(), nn.Linear(24, 3))
    # Load your trained weights here if needed:
    # model.load_state_dict(torch.load("weights.pt", weights_only=True))
    example = (torch.randn(1, 12),)
    validation = [(torch.randn(1, 12),) for _ in range(10)]
    return model, example, validation
```

```sh
python -m rvforge convert my_model.py --out build/custom --precision fp32
```

The factory may return `(model, example_inputs)` or `(model, example_inputs, validation_inputs)`. Inputs have fixed shapes and must be float32, int32 or int64. Integer token inputs use a C `int` interface and must stay within ±2²⁴. Output buffers are float32. The generated header gives the exact function signature; the manifest gives buffer sizes. Python files are executed as normal Python programs, so use model files you trust.

The Python API is also available:

```python
from rvforge import convert
manifest = convert(model, example, "build/export", precision="int8",
                   validation_inputs=validation)
```

Use an empty output directory. Conversion preserves an existing nonempty directory and returns an error. Export tracing warnings, unsupported operations and inconsistent verification shapes also produce errors.

## Compression choices

| Mode | Stored eligible weights | Arithmetic | Intended use |
|---|---|---|---|
| `fp32` | 32-bit floating point | FP32 | Establish the numerical baseline |
| `int8` | Signed 8-bit values and per-row scales | FP32 | Reduce constant storage with modest numerical drift |
| `binary` | Packed sign bits and mean-absolute row scales | FP32 | Explore aggressive lossy storage compression |

Eligible matrices are constant operands of dense multiplication, convolution and embedding lookup. A row follows the last logical tensor dimension. Biases, normalization parameters, activations and other constants remain floating point. Compressed weights are decoded as they are read; there is no full expanded-weight buffer.

This is **weight-storage compression**. It does not implement integer-only INT8 kernels or native one-bit arithmetic. Binary conversion performs no retraining. It can seriously reduce accuracy, including on a model that converts and runs correctly. Review the output drift in `manifest.json` and evaluate your own labeled validation data.

## Verification and evidence

The checks distinguish three questions:

1. Does graph export reproduce the Python model on the supplied inputs?
2. Does generated C reproduce an independent runtime using the same transformed weights?
3. Does compression preserve accuracy on the application's labeled data?

The first two are automatic. The third requires your task's labels. The included trained sequence benchmark demonstrates all three. It asks whether token 2 is greater than token 5 in a six-token sequence. Training and holdout sequences are disjoint. See [the validation report](VALIDATION.md) for measured results and [the operator reference](OPERATORS.md) for coverage.

```sh
python tests/run_matrix.py --zig /path/to/zig --riscv
python tests/test_edges.py --zig /path/to/zig --riscv
```

`tests/run_trained.py` additionally checks 1,000 held-out sequences per mode through compiled C. Its shared-library loading path is currently Windows-specific.

## Integration

Copy `model.c` and `model.h` into your C project. Allocate input and output buffers according to the manifest, call `rv_model_run`, and check its return value. Zero means success; `-2` is an invalid gather index, `-3` an unsupported integer range, `-4` a nonfinite input and `-5` a nonfinite intermediate result. Do not use outputs after an error. Input and output pointers must be non-null and correctly sized.

The workspace is static, so calls must be serialized. Shapes are fixed at conversion time. Generated C assumes IEEE-754 32-bit float and a 32-bit `int`. Math functions are freestanding approximations verified on the supplied numerical fixtures, not a general-purpose correctly rounded math library. There is no dynamic allocation or dependency on a runtime neural-network library in the generated application.

ONNX is used internally to describe the captured computation graph; ONNX Runtime supplies independent verification. Neither is required on the target device. Model-dependent Python branches, arbitrary Python code, recurrent control flow, training operations and unsupported kernels are outside this release. See [architecture](ARCHITECTURE.md) for extension points.

## License and attribution

RVForge source and the small internship example checkpoints are distributed under the MIT license. The RISC-V target harness is adapted from the internship's earlier inference project. PyTorch, NumPy, ONNX, ONNX Runtime, Zig and Node.js retain their own licenses. Third-party trained models retain their original licenses when converted. No third-party LLM weights are included in this repository.
