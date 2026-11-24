# Three minute demonstration

Start with the trained transformer example in INT8 mode. Explain the input task in one sentence: six token IDs enter the network, and it predicts whether the second token is greater than the fifth.

Show `examples/sequence_transformer.py`. It loads a normal PyTorch module and returns representative input tensors. Run the compilation and execution command in the [usage guide](USAGE.md) into a new empty directory. The command generates C, checks it on the host, creates the RISC-V executable and runs it in the emulator. Show the two PASS messages.

Open `model.h` to show the function that a C application calls, then `manifest.json` to show constant storage and working memory. Explain that INT8 compresses weight storage while arithmetic remains floating point in this release.

Next run `examples/cnn.py` or `examples/multi_input.py` through the same command. This demonstrates that the tool compiles a new combination of layers using the same pipeline. No model-specific C implementation is selected.

Finish with the measured task-accuracy table in `docs/VALIDATION.md`. Discuss what changed under binary conversion. A smaller file can still be a worse model; the tool exposes that tradeoff rather than hiding it.

Use the prebuilt executables as a fallback if installing Python packages or compiling is inconvenient during the presentation. State clearly that execution is in a functional RISC-V emulator. Physical hardware timing and deployment to arbitrary boards have not been established.
