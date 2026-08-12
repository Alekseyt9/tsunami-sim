"""CUDA regression for damage-driven procedural facade cracks."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

import numpy as np
import warp as wp

from kernels.hybrid import facade_crack_mask, facade_glass_coverage


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


@wp.kernel
def sample_glass_coverage(
    output: wp.array(dtype=float),
    panel: int,
    damage: float,
    side: int,
):
    index = wp.tid()
    x = index - (index // side) * side
    y = index // side
    uv = wp.vec2((float(x) + 0.5) / float(side), (float(y) + 0.5) / float(side))
    output[index] = facade_glass_coverage(panel, uv, damage)


def sample(device: str, panel: int, damage: float, material: int, side: int = 96) -> np.ndarray:
    output = wp.zeros(side * side, dtype=float, device=device)
    wp.launch(
        sample_crack_field, dim=side * side,
        inputs=[output, panel, damage, material, side], device=device,
    )
    wp.synchronize_device(device)
    return output.numpy()


def sample_coverage(device: str, panel: int, damage: float, side: int = 96) -> np.ndarray:
    output = wp.zeros(side * side, dtype=float, device=device)
    wp.launch(
        sample_glass_coverage, dim=side * side,
        inputs=[output, panel, damage, side], device=device,
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
    intact_coverage = sample_coverage(device, panel=17, damage=0.20)
    broken_coverage = sample_coverage(device, panel=17, damage=0.68)
    shattered_coverage = sample_coverage(device, panel=17, damage=1.0)
    if not np.all(intact_coverage == 1.0):
        raise AssertionError("intact glass contains premature holes")
    broken_missing = int(np.count_nonzero(broken_coverage < 0.5))
    shattered_missing = int(np.count_nonzero(shattered_coverage < 0.5))
    if broken_missing <= 0 or shattered_missing <= broken_missing:
        raise AssertionError(
            "glass holes do not grow progressively with damage: "
            f"{broken_missing} -> {shattered_missing}"
        )
    if shattered_missing >= int(0.98 * len(shattered_coverage)):
        raise AssertionError("shattered pane lost every edge/sliver instead of crumbling")
    print(
        f"PASS: glass cracks at 0.04 damage; concrete waits past 0.08 and produces "
        f"{visible}/{len(cracked_wall)} deterministic crack samples at 0.28 damage; "
        f"glass holes grow {broken_missing}->{shattered_missing} samples"
    )


if __name__ == "__main__":
    main()
