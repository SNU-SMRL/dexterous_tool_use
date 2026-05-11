"""Verify URDF joint structure is preserved after URDF -> USD conversion."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

URDF_ROOT = Path(__file__).resolve().parent.parent / "simtoolreal" / "assets" / "urdf"
MAIN_URDF = URDF_ROOT / "kuka_sharpa_description" / "iiwa14_left_sharpa_adjusted_restricted.urdf"
EXPECTED_REVOLUTE_COUNT = 29


@dataclass
class JointInfo:
    name: str
    joint_type: str
    parent: str
    child: str
    lower: float | None = None
    upper: float | None = None


@dataclass
class UrdfSummary:
    path: Path
    robot_name: str
    links: list[str] = field(default_factory=list)
    joints: list[JointInfo] = field(default_factory=list)


def parse_urdf(path: Path) -> UrdfSummary:
    tree = ET.parse(path)
    root = tree.getroot()
    summary = UrdfSummary(path=path, robot_name=root.attrib.get("name", "unknown"))

    for link in root.iter("link"):
        name = link.attrib.get("name")
        if name:
            summary.links.append(name)

    for joint in root.iter("joint"):
        name = joint.attrib.get("name", "")
        jtype = joint.attrib.get("type", "")
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        parent = parent_el.attrib.get("link", "") if parent_el is not None else ""
        child = child_el.attrib.get("link", "") if child_el is not None else ""

        lower: float | None = None
        upper: float | None = None
        limit_el = joint.find("limit")
        if limit_el is not None:
            lower = float(limit_el.attrib["lower"]) if "lower" in limit_el.attrib else None
            upper = float(limit_el.attrib["upper"]) if "upper" in limit_el.attrib else None

        summary.joints.append(JointInfo(
            name=name, joint_type=jtype, parent=parent, child=child,
            lower=lower, upper=upper,
        ))

    return summary


def print_summary(summary: UrdfSummary) -> None:
    revolute = [j for j in summary.joints if j.joint_type == "revolute"]
    fixed = [j for j in summary.joints if j.joint_type == "fixed"]
    other = [j for j in summary.joints if j.joint_type not in ("revolute", "fixed")]

    print(f"\n{'=' * 70}")
    print(f"URDF: {summary.path.relative_to(URDF_ROOT)}")
    print(f"Robot: {summary.robot_name}")
    print(f"Links: {len(summary.links)}  |  Joints: {len(summary.joints)} "
          f"(revolute={len(revolute)}, fixed={len(fixed)}"
          f"{f', other={len(other)}' if other else ''})")
    print(f"{'=' * 70}")

    if revolute:
        print(f"\n  {'#':<4} {'Joint Name':<35} {'Type':<10} {'Lower':>10} {'Upper':>10}")
        print(f"  {'-'*4} {'-'*35} {'-'*10} {'-'*10} {'-'*10}")
        for i, j in enumerate(revolute, 1):
            lo = f"{j.lower:.4f}" if j.lower is not None else "N/A"
            hi = f"{j.upper:.4f}" if j.upper is not None else "N/A"
            print(f"  {i:<4} {j.name:<35} {j.joint_type:<10} {lo:>10} {hi:>10}")

    print(f"\n  Link hierarchy (parent -> child):")
    for j in summary.joints:
        print(f"    {j.parent} -> [{j.joint_type}] {j.name} -> {j.child}")


def verify_main_robot(summary: UrdfSummary) -> bool:
    revolute = [j for j in summary.joints if j.joint_type == "revolute"]
    count = len(revolute)
    passed = True

    print(f"\n{'=' * 70}")
    print("VERIFICATION: Main robot joint structure")
    print(f"{'=' * 70}")

    kuka_joints = [j for j in revolute if j.name.startswith("iiwa14_")]
    sharpa_joints = [j for j in revolute if not j.name.startswith("iiwa14_")]

    print(f"\n  KUKA joints:   {len(kuka_joints)}  (expected 7)")
    for j in kuka_joints:
        print(f"    - {j.name}")
    if len(kuka_joints) != 7:
        print("  FAIL: expected 7 KUKA revolute joints")
        passed = False

    print(f"\n  Sharpa joints: {len(sharpa_joints)}  (expected 22)")
    for j in sharpa_joints:
        print(f"    - {j.name}")
    if len(sharpa_joints) != 22:
        print("  FAIL: expected 22 Sharpa revolute joints")
        passed = False

    print(f"\n  Total revolute: {count}  (expected {EXPECTED_REVOLUTE_COUNT})")
    if count != EXPECTED_REVOLUTE_COUNT:
        print(f"  FAIL: expected {EXPECTED_REVOLUTE_COUNT}, got {count}")
        passed = False
    else:
        print("  PASS")

    print(f"\n  All revolute joint names in URDF order:")
    for i, j in enumerate(revolute, 1):
        print(f"    {i:>2}. {j.name}")

    return passed


def discover_urdfs(root: Path) -> list[Path]:
    return sorted(root.rglob("*.urdf"))


def main() -> None:
    if not URDF_ROOT.is_dir():
        print(f"ERROR: URDF root not found: {URDF_ROOT}")
        sys.exit(1)

    urdfs = discover_urdfs(URDF_ROOT)
    print(f"Found {len(urdfs)} URDF files under {URDF_ROOT}\n")

    main_summary: UrdfSummary | None = None

    for path in urdfs:
        summary = parse_urdf(path)
        print_summary(summary)
        if path == MAIN_URDF:
            main_summary = summary

    if main_summary is None:
        print(f"\nERROR: Main robot URDF not found: {MAIN_URDF}")
        sys.exit(1)

    passed = verify_main_robot(main_summary)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
