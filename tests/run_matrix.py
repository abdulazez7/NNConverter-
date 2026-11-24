"""End-to-end differential tests across model families and storage precisions."""

import argparse, importlib.util, json, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rvforge import convert, ConversionError
from rvforge.verify import verify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zig", required=True)
    ap.add_argument("--out", type=Path, default=Path("builds/matrix"))
    ap.add_argument("--riscv", action="store_true")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    a.out.mkdir(parents=True, exist_ok=True)
    results = []
    for name in [
        "mlp",
        "cnn",
        "attention",
        "token_classifier",
        "multi_input",
        "signal_cnn",
    ]:
        spec = importlib.util.spec_from_file_location(
            name, root / "examples" / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for precision in ["fp32", "int8", "binary"]:
            folder = a.out / f"{name}-{precision}"
            start = time.perf_counter()
            try:
                model, example, validation = mod.build()
                meta = convert(model, example, folder, precision, validation)
                checked = verify(folder, a.zig, a.riscv)
                row = dict(
                    model=name,
                    precision=precision,
                    passed=True,
                    parameters=meta["parameter_storage_bytes"],
                    workspace=meta["workspace_bytes"],
                    operators=meta["operators"],
                    max_compression_error=max(
                        x["max_abs"] for x in meta["compression_output_error"]
                    ),
                    verification=checked,
                    seconds=round(time.perf_counter() - start, 2),
                )
            except Exception as e:
                row = dict(model=name, precision=precision, passed=False, error=str(e))
                print(type(e).__name__, str(e), flush=True)
            results.append(row)
            (a.out / "results.json").write_text(json.dumps(results, indent=2))
            print(name, precision, "PASS" if row["passed"] else "FAIL", flush=True)
    if not all(r["passed"] for r in results):
        raise SystemExit(1)
    print(
        f"ALL {len(results)} conversions passed; {len(results)*4} input fixtures per target",
        flush=True,
    )


if __name__ == "__main__":
    main()
