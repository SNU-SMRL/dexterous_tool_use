"""Generate procedural handle-head tool URDFs using SimToolReal's distribution.

Reuses SimToolReal's generate_objects.py and object_size_distributions.py
to create URDFs identical to those used during RL training.

Usage:
    source .venv/bin/activate
    python scripts/asset_conversion/generate_procedural_tools.py \
        --num-per-type 10 --output-dir assets/procedural_tools/urdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parents[2]
_simtoolreal = _project_root / "simtoolreal"

# Import directly to avoid isaacgym dependency from env.py
_gen_objects_path = _simtoolreal / "isaacgymenvs" / "tasks" / "simtoolreal"
sys.path.insert(0, str(_gen_objects_path))

from generate_objects import generate_handle_head_urdf
from object_size_distributions import OBJECT_SIZE_DISTRIBUTIONS

HANDLE_HEAD_TYPES = ["hammer", "screwdriver", "marker", "spatula", "eraser", "brush"]
OBJECT_BASE_SIZE = 0.04


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate procedural tool URDFs")
    parser.add_argument("--num-per-type", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="assets/procedural_tools/urdf")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    distributions = [d for d in OBJECT_SIZE_DISTRIBUTIONS if d.type in HANDLE_HEAD_TYPES]

    all_files: list[Path] = []
    all_scales: list[tuple[float, float, float]] = []
    manifest: list[dict] = []

    for dist in distributions:
        tool_type = dist.type
        handle_densities = dist.sample_handle_densities(args.num_per_type)
        head_densities = dist.sample_head_densities(args.num_per_type)
        handle_scales = dist.sample_handle_scales(args.num_per_type)
        head_scales = dist.sample_head_scales(args.num_per_type)

        for idx in range(args.num_per_type):
            name = f"{tool_type}_{idx:03d}"
            urdf_path = output_dir / f"{name}.urdf"

            generate_handle_head_urdf(
                filepath=urdf_path,
                handle_scale=handle_scales[idx],
                head_scale=head_scales[idx] if head_scales is not None else None,
                handle_density=handle_densities[idx],
                head_density=head_densities[idx] if head_scales is not None else None,
            )

            # Compute normalized scale for obs
            hs = handle_scales[idx]
            if len(hs) == 2:
                scale_3 = (hs[0], hs[1], hs[1])
            else:
                scale_3 = tuple(hs)
            norm_scale = tuple(s / OBJECT_BASE_SIZE for s in scale_3)

            all_files.append(urdf_path)
            all_scales.append(norm_scale)
            manifest.append({
                "name": name,
                "type": tool_type,
                "urdf": str(urdf_path),
                "scale": list(norm_scale),
                "handle_scale": list(handle_scales[idx]),
                "head_scale": list(head_scales[idx]) if head_scales is not None else None,
            })

    # Save manifest for later use
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {len(all_files)} URDFs in {output_dir}")
    print(f"Manifest: {manifest_path}")
    for t in HANDLE_HEAD_TYPES:
        count = sum(1 for m in manifest if m["type"] == t)
        print(f"  {t}: {count}")


if __name__ == "__main__":
    main()
