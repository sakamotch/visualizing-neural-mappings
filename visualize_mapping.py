"""
Visualize how randomly initialized 2D MLP layers warp the input space.

The script builds a grid of 2D points, assigns a smooth gradient color
to each point (for tracking), pushes the points through randomly
initialized 2x2 linear layers with a chosen activation, and plots the
result after each layer using the original colors. Figures can also be
saved to a specified output directory.
"""

import argparse
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


Activation = Callable[[np.ndarray], np.ndarray]


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def identity(x: np.ndarray) -> np.ndarray:
    return x


ACTIVATIONS: dict[str, Activation] = {
    "relu": relu,
    "tanh": np.tanh,
    "identity": identity,
}


def create_grid(
    grid_size: int, x_range: Tuple[float, float], y_range: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return grid points and a top-left -> bottom-right gradient color."""
    xs = np.linspace(x_range[0], x_range[1], grid_size)
    ys = np.linspace(y_range[0], y_range[1], grid_size)
    xv, yv = np.meshgrid(xs, ys)
    points = np.stack([xv.ravel(), yv.ravel()], axis=1)

    # Normalize to [0, 1] and blend x with inverted y to get a diagonal gradient.
    x_norm = (points[:, 0] - x_range[0]) / (x_range[1] - x_range[0])
    y_norm = (points[:, 1] - y_range[0]) / (y_range[1] - y_range[0])
    diag = 0.5 * (x_norm + (1 - y_norm))
    colors = plt.cm.viridis(diag)
    return points, colors


def build_layers(
    num_layers: int, weight_scale: float, bias_scale: float, rng: np.random.Generator
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create random 2x2 affine layers."""
    layers: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(num_layers):
        weight = rng.normal(scale=weight_scale, size=(2, 2))
        bias = rng.normal(scale=bias_scale, size=(2,))
        layers.append((weight, bias))
    return layers


def propagate(
    points: np.ndarray, layers: Iterable[Tuple[np.ndarray, np.ndarray]], activation: Activation
) -> List[np.ndarray]:
    """Return points after each layer (including the original points)."""
    outputs = [points]
    x = points
    for weight, bias in layers:
        x = activation(x @ weight.T + bias)
        outputs.append(x)
    return outputs


def plot_stages(
    stages: List[np.ndarray],
    colors: np.ndarray,
    activation_name: str,
    seed: int,
    output_dir: Optional[Path],
) -> List[Path]:
    """Scatter-plot points after each layer using original colors, saving one image per stage."""
    saved_paths: List[Path] = []
    if output_dir is None:
        output_dir = Path("outputs")
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
            f"mapping_activation-{activation_name}_layer-{idx}_of-{len(stages)-1}_seed-{seed}.png"
        )
        saved_path = output_dir / filename
        fig.savefig(saved_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(saved_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize how randomly initialized 2D MLP layers warp a grid of points. "
            "No training is performed."
        )
    )
    parser.add_argument("--grid-size", type=int, default=25, help="Number of grid points per axis")
    parser.add_argument(
        "--x-range", nargs=2, type=float, metavar=("X_MIN", "X_MAX"), default=(-1.0, 1.0)
    )
    parser.add_argument(
        "--y-range", nargs=2, type=float, metavar=("Y_MIN", "Y_MAX"), default=(-1.0, 1.0)
    )
    parser.add_argument("--layers", type=int, default=3, help="Number of 2D->2D layers")
    parser.add_argument(
        "--activation",
        choices=sorted(ACTIVATIONS.keys()),
        default="tanh",
        help="Activation function applied after every layer",
    )
    parser.add_argument("--weight-scale", type=float, default=1.0, help="Stddev for weights")
    parser.add_argument("--bias-scale", type=float, default=0.2, help="Stddev for biases")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save the plot PNG (created if missing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    points, colors = create_grid(args.grid_size, tuple(args.x_range), tuple(args.y_range))
    layers = build_layers(args.layers, args.weight_scale, args.bias_scale, rng)
    activation = ACTIVATIONS[args.activation]

    stages = propagate(points, layers, activation)
    saved_paths = plot_stages(stages, colors, args.activation, args.seed, args.output_dir)
    for path in saved_paths:
        print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
