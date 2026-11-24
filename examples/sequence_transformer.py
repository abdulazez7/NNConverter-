"""A trained internship model: is token 2 greater than token 5?"""

from pathlib import Path
import torch
from sequence_common import InferenceModel


def build():
    model = InferenceModel("transformer")
    model.load_state_dict(
        torch.load(
            Path(__file__).parent / "checkpoints/transformer_fp32.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    generator = torch.Generator().manual_seed(913)
    cases = [(torch.randint(0, 12, (1, 6), generator=generator),) for _ in range(8)]
    return model, cases[0], cases[1:]
