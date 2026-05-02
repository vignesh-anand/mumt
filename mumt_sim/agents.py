"""Add kinematic, visual-only Spot and humanoid articulated objects to the sim.

For M1 we explicitly do NOT depend on habitat-lab. The Spot standing-pose joint
values are constants copied from habitat-lab's
``habitat/articulated_agents/robots/spot_robot.py`` (``SpotRobot._get_spot_params``):
arm joints 0-6, gripper joint 7, four 3-joint legs 8-19 (hip yaw, hip pitch,
knee pitch). The robot is instantiated with ``MotionType.KINEMATIC`` and
``fixed_base=True`` so it does not fall under gravity, does not collide, and
does not have its joints simulated.

The humanoid URDFs are SMPL-X mannequins from habitat's ``humanoid_data``
bundle. They load in T-pose by default; SMPL-X motion playback is a milestone-2
concern.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import habitat_sim
import magnum as mn


SPOT_URDF: str = "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
HUMAN_URDF_TEMPLATE: str = "data/humanoids/humanoid_data/{name}/{name}.urdf"

# Standing pose joint values, ordered as the URDF declares them:
#   indices 0..6  : arm
#   index   7     : gripper
#   indices 8..19 : 4 legs * 3 joints (hip-yaw, hip-pitch, knee-pitch)
# Source: habitat-lab SpotRobot._get_spot_params(). These are public constants
# in habitat-lab; we copy the values here so M1 does not depend on lab at runtime.
_SPOT_ARM_INIT: tuple[float, ...] = (0.0, -3.14, 0.0, 3.0, 0.0, 0.0, 0.0)
_SPOT_GRIP_INIT: tuple[float, ...] = (-1.56,)
_SPOT_LEG_INIT: tuple[float, ...] = (0.0, 0.7, -1.5) * 4

SPOT_STANDING: tuple[float, ...] = _SPOT_ARM_INIT + _SPOT_GRIP_INIT + _SPOT_LEG_INIT
assert len(SPOT_STANDING) == 20

# Head camera offset in Spot's body frame, taken from
# ``SpotRobot._get_spot_params().cameras["articulated_agent_head_*"].cam_offset_pos``.
SPOT_HEAD_OFFSET: mn.Vector3 = mn.Vector3(0.479, 0.5, 0.0)


def _yaw_quat(yaw_rad: float) -> mn.Quaternion:
    """Return a quaternion representing a rotation by ``yaw_rad`` around world up (+Y)."""
    return mn.Quaternion.rotation(mn.Rad(yaw_rad), mn.Vector3.y_axis())


def _add_kinematic_articulated(
    sim: habitat_sim.Simulator,
    urdf_path: str,
    position: Sequence[float],
    yaw_rad: float,
    joint_positions: Optional[Iterable[float]] = None,
    fixed_base: bool = True,
):
    """Load a URDF as a kinematic articulated object and place it.

    Args:
        sim: habitat-sim Simulator that already loaded a scene.
        urdf_path: Path (relative to CWD) to the URDF file.
        position: World-frame XYZ to place the base at.
        yaw_rad: World-frame yaw (rad) around +Y for the base.
        joint_positions: Optional iterable of length == DOF. None leaves URDF defaults.
        fixed_base: Whether the base is welded to the world (True for visual-only).

    Returns:
        ``habitat_sim.physics.ManagedArticulatedObject`` (kinematic).
    """
    aom = sim.get_articulated_object_manager()
    ao = aom.add_articulated_object_from_urdf(urdf_path, fixed_base=fixed_base)
    if ao is None:
        raise RuntimeError(f"Failed to load URDF: {urdf_path}")

    # No physics on this body - we move it by setting transforms only.
    ao.motion_type = habitat_sim.physics.MotionType.KINEMATIC

    ao.translation = mn.Vector3(*position)
    ao.rotation = _yaw_quat(yaw_rad)

    if joint_positions is not None:
        jp = list(joint_positions)
        if len(jp) != len(ao.joint_positions):
            raise ValueError(
                f"joint_positions length {len(jp)} != AO DOF {len(ao.joint_positions)} "
                f"for URDF {urdf_path}"
            )
        ao.joint_positions = jp
    return ao


def add_kinematic_spot(
    sim: habitat_sim.Simulator,
    position: Sequence[float],
    yaw_rad: float = 0.0,
    urdf_path: str = SPOT_URDF,
):
    """Drop a kinematic Spot at ``position`` with the standing-pose joints set."""
    return _add_kinematic_articulated(
        sim,
        urdf_path=urdf_path,
        position=position,
        yaw_rad=yaw_rad,
        joint_positions=SPOT_STANDING,
        fixed_base=True,
    )


def add_kinematic_humanoid(
    sim: habitat_sim.Simulator,
    position: Sequence[float],
    yaw_rad: float = 0.0,
    name: str = "female_0",
    urdf_template: str = HUMAN_URDF_TEMPLATE,
):
    """Drop a kinematic humanoid mannequin (T-pose) at ``position``."""
    return _add_kinematic_articulated(
        sim,
        urdf_path=urdf_template.format(name=name),
        position=position,
        yaw_rad=yaw_rad,
        joint_positions=None,
        fixed_base=True,
    )
