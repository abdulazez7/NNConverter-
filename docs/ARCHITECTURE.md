# Architecture

The entry point receives an evaluation-mode PyTorch module and concrete input tensors. It exports a static computation graph, folds constant expressions, and records each tensor's shape. Operator handlers lower that graph to C loops using shapes and attributes. They do not branch on a network's model name.

For eligible constant matrices, the compiler optionally encodes per-row INT8 values or packed sign bits. Generated accessors reconstruct individual weights during FP32 arithmetic. The C program keeps fixed intermediate arrays and calls no external inference runtime.

Verification runs the original Python model, the exported floating-point graph, and a reference graph with transformed weights. The first comparison catches export discrepancies; the second supplies expected outputs for the generated C. Compression error is measured separately against the original graph. Numerical parity does not establish task accuracy or validity outside the supplied fixtures.

`rvforge/compiler.py` owns shape handling, constant encoding and operator lowering. `rvforge/verify.py` creates fixtures and orchestrates the host and RV32 builds. `rvforge/cli.py` loads a user's model factory. `rvforge/target/` supplies startup assembly, the QEMU-virt linker script and a functional RV32IMF emulator.

To add an operation, implement its shape and attribute constraints, lower its indexing and arithmetic, and add independent differential tests against Python and ONNX Runtime. Test different ranks, non-square dimensions, broadcasting and boundary values. Unsupported configurations should raise `ConversionError` before any final output files are written.

To add a board, supply its memory map, startup code, ABI, floating-point strategy and application I/O. Check constant storage, intermediate RAM and stack headroom. RV32IMF firmware cannot be assumed to run on an RV32IMC-only microcontroller. The generated C can instead be compiled by that board's toolchain, with software floating-point support if needed.

Future work includes live-range memory reuse, integer-only kernels with calibration, additional exporters, dynamic shapes, independently executed QEMU CI and hardware timing. These are extension directions, not current features.
