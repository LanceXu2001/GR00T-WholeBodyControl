#!/usr/bin/env python3
"""Visualize CNN height-map input ``(5, 15, 15)`` as five 3D scatter subplots.

Tensor index ``[t, i, j]`` is placed at world-like coordinates:
    x = i * resolution
    y = j * resolution
    z = height_map[t, i, j]

so ``[0, 0]`` → ``(0, 0, h)``, ``[0, 1]`` → ``(0, 0.1, h)``, etc.

Default npy path matches the deploy dump:
    gear_sonic_deploy/height_map_2d_history5.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


def load_height_map(path: Path) -> np.ndarray:
    height_map = np.load(path)
    if height_map.ndim != 3:
        raise ValueError(f"Expected (T, H, W), got shape {height_map.shape}")
    return height_map.astype(np.float64, copy=False)


def plot_height_map_history(
    height_map: np.ndarray,
    *,
    resolution: float = 0.1,
    title: str | None = None,
) -> plt.Figure:
    num_frames, height_count, width_count = height_map.shape
    row_indices = np.arange(height_count, dtype=np.float64)
    col_indices = np.arange(width_count, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(row_indices * resolution, col_indices * resolution, indexing="ij")

    z_min = float(np.nanmin(height_map))
    z_max = float(np.nanmax(height_map))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
        z_min, z_max = -0.5, 0.5

    fig = plt.figure(figsize=(4 * num_frames, 4.5))
    if title:
        fig.suptitle(title, fontsize=12)

    for frame_index in range(num_frames):
        axis = fig.add_subplot(1, num_frames, frame_index + 1, projection="3d")
        heights = height_map[frame_index]
        scatter = axis.scatter(
            grid_x.ravel(),
            grid_y.ravel(),
            heights.ravel(),
            c=heights.ravel(),
            cmap="viridis",
            vmin=z_min,
            vmax=z_max,
            s=18,
            depthshade=True,
        )
        axis.set_title(f"frame {frame_index}" + (" (oldest)" if frame_index == 0 else "")
                       + (" (newest)" if frame_index == num_frames - 1 else ""))
        axis.set_xlabel("i * res (m)")
        axis.set_ylabel("j * res (m)")
        axis.set_zlabel("height (m)")
        axis.set_xlim(0.0, (height_count - 1) * resolution)
        axis.set_ylim(0.0, (width_count - 1) * resolution)
        axis.set_zlim(z_min, z_max)
        fig.colorbar(scatter, ax=axis, shrink=0.55, pad=0.08, label="height")

    fig.tight_layout()
    return fig


def main() -> None:
    default_npy = Path(__file__).resolve().parent / "height_map_2d_history5.npy"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npy",
        type=Path,
        default=default_npy,
        help=f"Path to (T,H,W) npy (default: {default_npy})",
    )
    parser.add_argument("--resolution", type=float, default=0.1, help="Grid spacing in meters.")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the figure (e.g. height_map_vis.png).",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window.")
    args = parser.parse_args()

    if not args.npy.is_file():
        raise FileNotFoundError(f"Height-map npy not found: {args.npy}")

    height_map = load_height_map(args.npy)
    print(f"Loaded {args.npy} shape={height_map.shape} "
          f"min={height_map.min():.4f} max={height_map.max():.4f}")

    figure = plot_height_map_history(
        height_map,
        resolution=args.resolution,
        title=str(args.npy),
    )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {args.save}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
