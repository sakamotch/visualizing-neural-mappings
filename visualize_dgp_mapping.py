"""
Visualize how a randomly sampled deep Gaussian process (DGP) warps the 2D input space.

A grid of colored 2D points is passed through multiple GP layers (sampled from an
exact RBF kernel). Each layer's output is saved as an individual image.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def create_grid(
    grid_size: int, x_range: Tuple[float, float], y_range: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(x_range[0], x_range[1], grid_size)
    ys = np.linspace(y_range[0], y_range[1], grid_size)
    xv, yv = np.meshgrid(xs, ys)
    points = np.stack([xv.ravel(), yv.ravel()], axis=1)

    x_norm = (points[:, 0] - x_range[0]) / (x_range[1] - x_range[0])
    y_norm = (points[:, 1] - y_range[0]) / (y_range[1] - y_range[0])
    diag = 0.5 * (x_norm + (1 - y_norm))
    colors = plt.cm.viridis(diag)
    return points, colors


def rbf_covariance(inputs: np.ndarray, lengthscale: float, output_scale: float, jitter: float) -> np.ndarray:
    """Compute exact RBF kernel matrix with added jitter for stability."""
    diff_sq = (
        np.sum(inputs**2, axis=1)[:, None]
        + np.sum(inputs**2, axis=1)[None, :]
        - 2 * inputs @ inputs.T
    )
    K = (output_scale**2) * np.exp(-0.5 * diff_sq / (lengthscale**2))
    K += jitter * np.eye(inputs.shape[0])
    return K


def sample_gp_layer(
    inputs: np.ndarray,
    rng: np.random.Generator,
    lengthscale: float,
    output_scale: float,
    jitter: float,
) -> np.ndarray:
    """
    Sample a GP mapping R^2 -> R^2 using an exact RBF kernel.
    """
    K = rbf_covariance(inputs, lengthscale=lengthscale, output_scale=output_scale, jitter=jitter)
    mean = np.zeros(inputs.shape[0])
    # Two independent output dimensions.
    samples = rng.multivariate_normal(mean=mean, cov=K, size=2)  # (2, N)
    return samples.T  # (N, 2)


def propagate_dgp(
    points: np.ndarray,
    layers: int,
    lengthscale: float,
    output_scale: float,
    jitter: float,
    seed: int,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    outputs = [points]
    x = points
    for _ in range(layers):
        x = sample_gp_layer(x, rng, lengthscale, output_scale, jitter)
        outputs.append(x)
    return outputs


def save_layers(
    stages: List[np.ndarray],
    colors: np.ndarray,
    lengthscale: float,
    output_scale: float,
    seed: int,
    output_dir: Optional[Path],
) -> List[Path]:
    saved_paths: List[Path] = []
    if output_dir is None:
        output_dir = Path("outputs_dgp")
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, pts in enumerate(stages):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=8, linewidths=0)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.6)
        ax.axvline(0, color="gray", lw=0.5, alpha=0.6)
        ax.set_aspect("equal")
        ax.set_title("Input" if idx == 0 else f"After layer {idx}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.tight_layout()

        filename = (
            f"mapping_dgp_layer-{idx}_of-{len(stages)-1}"
            f"_ls-{lengthscale}"
            f"_scale-{output_scale}"
            f"_seed-{seed}.png"
        )
        saved_path = output_dir / filename
        fig.savefig(saved_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(saved_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize how a randomly sampled deep Gaussian process warps a 2D grid. "
            "Each layer is a GP sample approximated with random Fourier features."
        )
    )
    parser.add_argument("--grid-size", type=int, default=25, help="Number of grid points per axis")
    parser.add_argument(
        "--x-range", nargs=2, type=float, metavar=("X_MIN", "X_MAX"), default=(-1.0, 1.0)
    )
    parser.add_argument(
        "--y-range", nargs=2, type=float, metavar=("Y_MIN", "Y_MAX"), default=(-1.0, 1.0)
    )
    parser.add_argument("--layers", type=int, default=3, help="Number of GP layers")
    parser.add_argument(
        "--lengthscale",
        type=float,
        default=1.0,
        help="RBF kernel lengthscale (smaller -> more wiggly functions)",
    )
    parser.add_argument(
        "--output-scale",
        type=float,
        default=1.0,
        help="Stddev for the GP output weights (scales layer outputs)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--jitter",
        type=float,
        default=1e-6,
        help="Diagonal jitter added to the kernel for numerical stability",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_dgp"),
        help="Directory to save per-layer PNGs (created if missing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points, colors = create_grid(args.grid_size, tuple(args.x_range), tuple(args.y_range))

    stages = propagate_dgp(
        points,
        layers=args.layers,
        lengthscale=args.lengthscale,
        output_scale=args.output_scale,
        jitter=args.jitter,
        seed=args.seed,
    )

    saved_paths = save_layers(
        stages=stages,
        colors=colors,
        lengthscale=args.lengthscale,
        output_scale=args.output_scale,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    for path in saved_paths:
        print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
