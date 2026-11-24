"""Compile the real internship checkpoints and evaluate C on 1000 held-out sequences."""

import argparse, ctypes, importlib.util, json, subprocess, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
from rvforge import convert
from rvforge.verify import verify


def dataset(seed, count, excluded=()):
    rng = np.random.default_rng(seed)
    seen = set(excluded)
    xs = []
    while len(xs) < count:
        seq = tuple(rng.choice(12, 6, replace=False))
        if seq in seen:
            continue
        seen.add(seq)
        xs.append(seq)
    x = np.array(xs, dtype=np.int64)
    return x, (x[:, 1] > x[:, 4]).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zig", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "builds/trained")
    a = ap.parse_args()
    zig = str(Path(a.zig).resolve())
    a.out.mkdir(parents=True, exist_ok=True)
    results = []
    torch.set_num_threads(1)
    train, _ = dataset(112, 3500)
    x, y = dataset(913, 1000, map(tuple, train.tolist()))
    for name in ["mlp", "transformer"]:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "examples" / f"sequence_{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for precision in ["fp32", "int8", "binary"]:
            model, example, validation = mod.build()
            folder = a.out / f"{name}-{precision}"
            meta = convert(model, example, folder, precision, validation)
            check = verify(folder, zig, True)
            dll = folder / "model.dll"
            subprocess.run(
                [
                    zig,
                    "cc",
                    "-w",
                    "-shared",
                    "-O2",
                    "-ffp-contract=off",
                    "-Wl,--export-all-symbols",
                    str(folder / "model.c"),
                    "-o",
                    str(dll),
                ],
                check=True,
                capture_output=True,
            )
            lib = ctypes.CDLL(str(dll.resolve()))
            fn = lib.rv_model_run
            fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)]
            fn.restype = ctypes.c_int
            predictions = []
            for seq in x:
                inp = np.ascontiguousarray(seq, dtype=np.int32)
                out = np.zeros(2, dtype=np.float32)
                status = fn(
                    inp.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                )
                if status:
                    raise RuntimeError(f"Inference status {status}")
                predictions.append(int(np.argmax(out)))
            accuracy = float(np.mean(np.array(predictions) == y))
            row = dict(
                model=name,
                precision=precision,
                correct=int(np.sum(np.array(predictions) == y)),
                total=len(y),
                accuracy=accuracy,
                parameter_storage_bytes=meta["parameter_storage_bytes"],
                workspace_bytes=meta["workspace_bytes"],
                verification=check,
            )
            results.append(row)
            (a.out / "results.json").write_text(json.dumps(results, indent=2))
            print(name, precision, row["correct"], "/1000", flush=True)
    print(
        "COMPLETE six trained conversions, 48 fixtures per target and 6000 C classifications",
        flush=True,
    )


if __name__ == "__main__":
    main()
