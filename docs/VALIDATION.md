# Release validation

Validation date: 22 November 2026. The measured target is an RV32IMF functional emulator using the QEMU `virt` memory map. Tests were run on Windows with Python 3.11, Zig 0.16.0, PyTorch, ONNX Runtime and Node.js. Exact package versions accompany the release.

## End-to-end checks

- Six example network families × three storage modes: **18 conversions passed**.
- Four fixtures per conversion: **72 fixtures passed on the host and 72 on RV32**.
- Edge and rejection suite: **22 tests passed**, including broadcasting, sliced tensors, cropping and padding, multi-axis normalization, grouped convolution, reduction axes, pooling padding, nonlinear functions, unseen layer sizes and unsafe export rejection.
- Two trained internship models × three modes: **six additional conversions passed**, with **48 fixtures on each target** and **6,000 labeled classifications through compiled C**.
- A wheel built from the release was installed into a separate directory. Its CLI converted the trained transformer in INT8 mode and passed eight host and eight RV32 numerical fixtures, confirming the package includes its target support files.

All generated output elements are compared with independent reference outputs using absolute error ≤ 0.0003 + 0.0003 × |reference|. The original PyTorch module is also checked against its exported graph. Weight-compression drift is recorded separately in each manifest.

## Trained task accuracy

The task predicts whether the second token exceeds the fifth in a sequence of six distinct token IDs drawn from 0 to 11. The holdout contains 1,000 sequences generated with seed 913, excluding the 3,500 training sequences generated with seed 112. These are small synthetic models; the results do not establish performance on real-world data.

| Model | Storage mode | Correct out of 1000 | Accuracy |
|---|---|---:|---:|
| MLP | FP32 | 1000 | 100.0% |
| MLP | INT8 weights | 1000 | 100.0% |
| MLP | Binary weights | 850 | 85.0% |
| Transformer | FP32 | 1000 | 100.0% |
| Transformer | INT8 weights | 1000 | 100.0% |
| Transformer | Binary weights | 838 | 83.8% |

The public converter applies post-training compression to eligible dense and embedding matrices. It does not reuse the earlier project's binary-aware retraining process. These results therefore should not be compared as though they were the same binary models. Binary storage saves space but loses useful information; the INT8 demonstrations retain accuracy on this test.

## Practical limits

The included examples establish conversion and numerical correctness for exercised shapes and operations. Unseen models still need their own representative fixtures and labeled evaluation. No physical-board run, cycle-accurate measurement, energy measurement, peak-RAM profiling or independent QEMU run is claimed. Static workspace and parameter byte counts describe generated arrays, not total firmware or runtime memory.

The release contains CI configuration, but no remote GitHub Actions run has occurred yet. Local test result files are the evidence for this release.
