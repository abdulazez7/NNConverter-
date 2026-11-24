"""Python-first command line. Model files are trusted Python programs."""

import argparse, importlib.util, json, sys
from pathlib import Path
from .compiler import convert, ConversionError
from .verify import verify


def main():
    ap = argparse.ArgumentParser(
        description="Convert a Python neural network to portable C and optional RV32 firmware."
    )
    sub = ap.add_subparsers(dest="command", required=True)
    cv = sub.add_parser("convert")
    cv.add_argument("model", type=Path)
    cv.add_argument("--factory", default="build")
    cv.add_argument("--precision", choices=["fp32", "int8", "binary"], default="fp32")
    cv.add_argument("--out", type=Path, required=True)
    cv.add_argument("--zig", type=Path)
    cv.add_argument("--riscv", action="store_true")
    cv.add_argument("--node", default="node")
    args = ap.parse_args()
    try:
        if args.riscv and not args.zig:
            raise ConversionError("--riscv requires --zig pointing to Zig 0.16")
        modelpath = args.model.resolve()
        sys.path.insert(0, str(modelpath.parent))
        spec = importlib.util.spec_from_file_location("rvforge_user_model", modelpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        returned = getattr(mod, args.factory)()
        if not isinstance(returned, tuple) or len(returned) not in (2, 3):
            raise ConversionError(
                "Factory must return (model, example_inputs) or (model, example_inputs, validation_inputs)"
            )
        model, examples = returned[:2]
        validation = returned[2] if len(returned) == 3 else None
        report = convert(model, examples, args.out, args.precision, validation)
        print(
            f'Created {args.out.resolve()} | {report["operator_count"]} operations | {report["parameter_storage_bytes"]} parameter bytes | {report["workspace_bytes"]} workspace bytes'
        )
        if args.zig:
            print(
                json.dumps(verify(args.out, args.zig, args.riscv, args.node), indent=2)
            )
        if args.precision != "fp32":
            print(
                "Compression changes model outputs. Review manifest.json and evaluate task accuracy before deployment."
            )
    except (
        ConversionError,
        RuntimeError,
        ValueError,
        AttributeError,
        FileNotFoundError,
    ) as e:
        print("Conversion failed: " + str(e), file=sys.stderr)
        return 2
    return 0
