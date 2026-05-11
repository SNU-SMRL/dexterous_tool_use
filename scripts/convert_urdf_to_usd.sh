#!/usr/bin/env bash
# Batch-convert SimToolReal URDF assets to USD via Isaac Lab's convert_urdf.py.
# Usage: bash scripts/convert_urdf_to_usd.sh
#
# Requires Isaac Sim runtime (launched internally by convert_urdf.py).

set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONVERT_SCRIPT="${PROJ_ROOT}/IsaacLab/scripts/tools/convert_urdf.py"
URDF_ROOT="${PROJ_ROOT}/simtoolreal/assets/urdf"
USD_ROOT="${PROJ_ROOT}/assets/usd"

mkdir -p "${USD_ROOT}"

convert() {
    local urdf_path="$1"
    local usd_path="$2"
    shift 2
    local extra_args=("$@")

    if [[ -f "${usd_path}" ]]; then
        echo "[SKIP] Already exists: ${usd_path}"
        return
    fi

    mkdir -p "$(dirname "${usd_path}")"
    echo "[CONVERT] ${urdf_path} -> ${usd_path}"
    python "${CONVERT_SCRIPT}" \
        "${urdf_path}" "${usd_path}" \
        --headless \
        "${extra_args[@]}"
}

echo "=== 1. Robot: KUKA iiwa14 + Sharpa Hand ==="
convert \
    "${URDF_ROOT}/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf" \
    "${USD_ROOT}/robot/kuka_sharpa.usd" \
    --fix-base

echo ""
echo "=== 2. Tools (6 categories x 2 variants) ==="
for category in brush eraser hammer marker screwdriver spatula; do
    for urdf in "${URDF_ROOT}/dextoolbench/${category}"/*/*.urdf; do
        variant="$(basename "$(dirname "${urdf}")")"
        convert "${urdf}" "${USD_ROOT}/tools/${category}/${variant}.usd" --merge-joints
    done
done

echo ""
echo "=== 3. Tables ==="
for urdf in "${URDF_ROOT}"/table_narrow*.urdf; do
    name="$(basename "${urdf}" .urdf)"
    convert "${urdf}" "${USD_ROOT}/tables/${name}.usd" --fix-base --merge-joints
done

echo ""
echo "=== 4. Environment objects (task-specific) ==="
for urdf in "${URDF_ROOT}"/dextoolbench/environments/*/*/*.urdf; do
    rel="${urdf#${URDF_ROOT}/dextoolbench/environments/}"
    # e.g. hammer/claw_hammer/swing_down.urdf -> environments/hammer/claw_hammer/swing_down.usd
    usd_rel="${rel%.urdf}.usd"
    convert "${urdf}" "${USD_ROOT}/environments/${usd_rel}" --fix-base --merge-joints
done

echo ""
echo "=== Done ==="
echo "USD assets written to: ${USD_ROOT}"
find "${USD_ROOT}" -name "*.usd" | sort
