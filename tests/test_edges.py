"""Differential edge cases and failures that must not silently compile."""

import argparse, json, sys, tempfile, unittest
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rvforge import convert, ConversionError
from rvforge.verify import verify

ZIG = None
RISC = False
RESULTS = []


class Function(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *xs):
        return self.fn(*xs)


class Edges(unittest.TestCase):
    def check(self, label, model, inputs, precision="fp32"):
        with tempfile.TemporaryDirectory(prefix="rvforge-test-") as td:
            out = Path(td) / "build"
            meta = convert(model, inputs, out, precision)
            result = verify(out, ZIG, RISC)
            RESULTS.append(
                dict(
                    case=label,
                    precision=precision,
                    operators=meta["operators"],
                    result=result,
                )
            )

    def test_batched_matmul_broadcast(self):
        self.check(
            "batched_matmul_broadcast",
            Function(lambda a, b: a @ b),
            (torch.randn(2, 1, 3, 4), torch.randn(1, 5, 4, 6)),
        )

    def test_elementwise_broadcast(self):
        self.check(
            "broadcast",
            Function(lambda a, b: (a + b) * (a - b) / (b.abs() + 1)),
            (torch.randn(2, 3, 4), torch.randn(1, 4)),
        )

    def test_slice_concat_split(self):
        self.check(
            "slice_concat_split",
            Function(lambda x: torch.cat(torch.split(x[:, 1:7:2], 1, dim=1), dim=2)),
            (torch.randn(2, 8, 3),),
        )

    def test_pad_crop(self):
        self.check(
            "pad_crop",
            Function(lambda x: F.pad(x, (-1, 2, 2, 1), "constant", 0.5)),
            (torch.randn(2, 3, 4, 5),),
        )

    def test_nonlast_softmax(self):
        self.check(
            "softmax_axis",
            Function(lambda x: (x.softmax(1), x.log_softmax(2))),
            (torch.randn(2, 3, 4),),
        )

    def test_reductions(self):
        self.check(
            "reductions",
            Function(
                lambda x: (x.sum((0, 2)), x.mean((1, 2), keepdim=True), x.amax(1))
            ),
            (torch.randn(2, 3, 4),),
        )

    def test_norm_multiple_dimensions(self):
        self.check("layernorm_multiaxis", nn.LayerNorm((3, 4)), (torch.randn(2, 3, 4),))

    def test_conv_group_stride_dilation(self):
        self.check(
            "group_stride_dilation",
            nn.Conv2d(
                4, 6, (2, 3), stride=(2, 1), padding=(1, 2), dilation=(2, 1), groups=2
            ),
            (torch.randn(2, 4, 8, 7),),
            "int8",
        )

    def test_pool_padding(self):
        self.check(
            "pool_padding",
            Function(
                lambda x: (
                    F.avg_pool2d(x, 3, 2, 1, count_include_pad=False),
                    F.avg_pool2d(x, 3, 2, 1, count_include_pad=True),
                    F.max_pool2d(x, 3, 2, 1),
                )
            ),
            (torch.randn(2, 3, 8, 7),),
        )

    def test_nonlinear(self):
        self.check(
            "nonlinear",
            Function(
                lambda x: (
                    F.elu(x),
                    F.softplus(x),
                    x.exp(),
                    (x.abs() + 0.1).log(),
                    (x * x + 0.1).sqrt(),
                    x.clamp(-0.5, 0.5),
                )
            ),
            (torch.randn(3, 7),),
        )

    def test_extreme_exp(self):
        self.check(
            "exp_subnormal",
            Function(lambda x: x.exp()),
            (torch.tensor([-100.0, -90.0, -87.0, -1.0, 0.0, 1.0, 20.0, 80.0]),),
        )

    def test_masks(self):
        self.check(
            "where_masks",
            Function(lambda x: torch.where(x > 0, x, x * 0.2)),
            (torch.randn(2, 5),),
        )

    def test_expand(self):
        self.check(
            "expand",
            Function(lambda x: x.unsqueeze(1).expand(2, 3, 4)),
            (torch.randn(2, 4),),
        )

    def test_zero_weights(self):
        m = nn.Linear(5, 3)
        nn.init.zeros_(m.weight)
        for precision in ["int8", "binary"]:
            self.check("zero_weights", m, (torch.randn(2, 5),), precision)

    def test_unseen_dimensions(self):
        torch.manual_seed(77)
        for a, b, c in [(3, 11, 7), (13, 5, 2), (1, 1, 1)]:
            m = nn.Sequential(nn.Linear(a, b), nn.ReLU(), nn.Linear(b, c))
            self.check(f"unseen_mlp_{a}_{b}_{c}", m, (torch.randn(3, a),), "binary")

    def test_reject_unsupported(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ConversionError, "Unsupported operator Sin"):
                convert(Function(torch.sin), torch.randn(2, 3), td)
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_reject_data_dependent_branch(self):
        def branch(x):
            return x * 2 if x.sum() > 0 else x - 2

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ConversionError, "graph export failed"):
                convert(Function(branch), torch.ones(2, 3), td)

    def test_reject_bad_dtype(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ConversionError, "dtype"):
                convert(nn.ReLU(), torch.ones(2, dtype=torch.float64), td)

    def test_reject_integer_arithmetic(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ConversionError, "integer arithmetic"):
                convert(Function(lambda x: x + 1), torch.tensor([1, 2]), td)

    def test_reject_changed_shape(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ConversionError, "shapes"):
                convert(
                    nn.ReLU(),
                    torch.ones(2, 3),
                    td,
                    validation_inputs=[torch.ones(4, 3)],
                )

    def test_reject_out_of_range_integer(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ConversionError, "Integer inputs"):
                convert(nn.Embedding(2, 2), torch.tensor([2**25]), td)

    def test_reject_existing_output(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "keep.txt"
            p.write_text("unchanged")
            with self.assertRaisesRegex(ConversionError, "empty"):
                convert(nn.ReLU(), torch.ones(2), td)
            self.assertEqual(p.read_text(), "unchanged")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zig", required=True)
    ap.add_argument("--riscv", action="store_true")
    ap.add_argument("--report", default="builds/edge-results.json")
    a, rest = ap.parse_known_args()
    ZIG = a.zig
    RISC = a.riscv
    torch.set_num_threads(1)
    torch.manual_seed(46)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Edges)
    )
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(
        json.dumps(
            dict(
                tests=result.testsRun,
                failures=len(result.failures),
                errors=len(result.errors),
                differential_cases=RESULTS,
            ),
            indent=2,
        )
    )
    raise SystemExit(not result.wasSuccessful())
