"""CUDA regression for damage-driven procedural facade cracks."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
import warp as wp

from kernels.hybrid import facade_crack_mask


@wp.kernel
def sample_crack_field(
    output: wp.array(dtype=float),
    panel: int,
    damage: float,
    material: int,
    side: int,
):
    index = wp.tid()
    x = index - (index // side) * side
    y = index // side
    uv = wp.vec2((float(x) + 0.5) / float(side), (float(y) + 0.5) / float(side))
    output[index] = facade_crack_mask(panel, uv, damage, material)


def sample(device: str, panel: int, damage: float, material: int, side: int = 96) -> np.ndarray:
    output = wp.zeros(side * side, dtype=float, device=device)
    wp.launch(
        sample_crack_field, dim=side * side,
        inputs=[output, panel, damage, material, side], device=device,
    )
    wp.synchronize_device(device)
    return output.numpy()


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    intact_wall = sample(device, panel=17, damage=0.04, material=10)
    early_glass = sample(device, panel=17, damage=0.04, material=20)
    cracked_wall = sample(device, panel=17, damage=0.28, material=10)
    repeated_wall = sample(device, panel=17, damage=0.28, material=10)
    if float(np.max(intact_wall)) != 0.0:
        raise AssertionError("concrete showed cracks below its visibility threshold")
    if int(np.count_nonzero(early_glass > 0.05)) == 0:
        raise AssertionError("brittle glass did not crack before concrete")
    visible = int(np.count_nonzero(cracked_wall > 0.05))
    if visible == 0 or visible > int(0.30 * len(cracked_wall)):
        raise AssertionError(f"procedural concrete crack coverage is implausible: {visible} samples")
    if not np.array_equal(cracked_wall, repeated_wall):
        raise AssertionError("facade crack pattern is not deterministic")
    print(
        f"PASS: glass cracks at 0.04 damage; concrete waits past 0.08 and produces "
        f"{visible}/{len(cracked_wall)} deterministic crack samples at 0.28 damage"
    )


if __name__ == "__main__":
    main()
