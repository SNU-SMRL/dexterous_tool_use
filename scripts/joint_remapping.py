"""Joint order remapping between IsaacGym (depth-first) and Isaac Lab (breadth-first).

Derived from URDF: iiwa14_left_sharpa_adjusted_restricted.urdf
Robot: 7-DOF KUKA iiwa14 arm + 22-DOF Sharpa hand = 29 revolute DOFs.
"""

from __future__ import annotations

import torch

JOINT_NAMES_ISAACGYM: list[str] = [
    "iiwa14_joint_1",
    "iiwa14_joint_2",
    "iiwa14_joint_3",
    "iiwa14_joint_4",
    "iiwa14_joint_5",
    "iiwa14_joint_6",
    "iiwa14_joint_7",
    "left_1_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_2_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_3_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_4_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_5_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
]

JOINT_NAMES_ISAACLAB: list[str] = [
    "iiwa14_joint_1",
    "iiwa14_joint_2",
    "iiwa14_joint_3",
    "iiwa14_joint_4",
    "iiwa14_joint_5",
    "iiwa14_joint_6",
    "iiwa14_joint_7",
    "left_1_thumb_CMC_FE",
    "left_2_index_MCP_FE",
    "left_3_middle_MCP_FE",
    "left_4_ring_MCP_FE",
    "left_5_pinky_CMC",
    "left_thumb_CMC_AA",
    "left_index_MCP_AA",
    "left_middle_MCP_AA",
    "left_ring_MCP_AA",
    "left_pinky_MCP_FE",
    "left_thumb_MCP_FE",
    "left_index_PIP",
    "left_middle_PIP",
    "left_ring_PIP",
    "left_pinky_MCP_AA",
    "left_thumb_MCP_AA",
    "left_index_DIP",
    "left_middle_DIP",
    "left_ring_DIP",
    "left_pinky_PIP",
    "left_thumb_IP",
    "left_pinky_DIP",
]

NUM_DOFS: int = 29

_isaaclab_name_to_idx = {name: i for i, name in enumerate(JOINT_NAMES_ISAACLAB)}
_isaacgym_name_to_idx = {name: i for i, name in enumerate(JOINT_NAMES_ISAACGYM)}

ISAACGYM_TO_ISAACLAB: list[int] = [
    _isaaclab_name_to_idx[name] for name in JOINT_NAMES_ISAACGYM
]

ISAACLAB_TO_ISAACGYM: list[int] = [
    _isaacgym_name_to_idx[name] for name in JOINT_NAMES_ISAACLAB
]


def remap_tensor(
    tensor: torch.Tensor,
    mapping: list[int],
    dim: int = -1,
) -> torch.Tensor:
    """Reorder ``tensor`` along ``dim`` according to ``mapping``.

    Args:
        tensor: Input tensor with size ``NUM_DOFS`` along ``dim``.
        mapping: Index list where ``mapping[i]`` is the source index for
            position ``i`` in the output.  Use ``ISAACGYM_TO_ISAACLAB`` to
            convert IsaacGym-ordered data to Isaac Lab order, or
            ``ISAACLAB_TO_ISAACGYM`` for the reverse.
        dim: Dimension to reorder.
    """
    idx = torch.tensor(mapping, dtype=torch.long, device=tensor.device)
    return tensor.index_select(dim, idx)


if __name__ == "__main__":
    print(f"{'Idx':>3}  {'IsaacGym joint':^30}  {'->':^4}  {'Isaac Lab idx':>13}")
    print("-" * 60)
    for ig_idx, name in enumerate(JOINT_NAMES_ISAACGYM):
        il_idx = ISAACGYM_TO_ISAACLAB[ig_idx]
        print(f"{ig_idx:>3}  {name:<30}  {'->':^4}  {il_idx:>13}")

    print()
    print(f"{'Idx':>3}  {'Isaac Lab joint':^30}  {'->':^4}  {'IsaacGym idx':>13}")
    print("-" * 60)
    for il_idx, name in enumerate(JOINT_NAMES_ISAACLAB):
        ig_idx = ISAACLAB_TO_ISAACGYM[il_idx]
        print(f"{il_idx:>3}  {name:<30}  {'->':^4}  {ig_idx:>13}")

    print("\nISAACGYM_TO_ISAACLAB =", ISAACGYM_TO_ISAACLAB)
    print("ISAACLAB_TO_ISAACGYM =", ISAACLAB_TO_ISAACGYM)

    # Round-trip sanity check
    t = torch.arange(NUM_DOFS, dtype=torch.float32)
    t_lab = remap_tensor(t, ISAACGYM_TO_ISAACLAB)
    t_gym = remap_tensor(t_lab, ISAACLAB_TO_ISAACGYM)
    assert torch.equal(t, t_gym), "Round-trip failed"
    print("\nRound-trip check passed.")
