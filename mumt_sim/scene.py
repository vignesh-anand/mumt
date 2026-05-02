"""Build the habitat-sim Simulator and configure the multi-agent camera rig.

Three habitat-sim Agents are allocated:
  - agent 0: ``observer``      - free-cam, RGB only, used for the third-person sanity render
  - agent 1: ``spot_0_head``   - RGB + depth, pose driven by ``mumt_sim.pan_tilt.PanTiltHead``
  - agent 2: ``spot_1_head``   - RGB + depth, pose driven by ``mumt_sim.pan_tilt.PanTiltHead``

The Spot articulated objects themselves are added later via ``mumt_sim.agents``;
this module only builds the camera rig and loads the scene.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import habitat_sim


# Sensor uuids per Spot. habitat-sim's internal sensor registry is keyed
# globally by UUID, so each agent must use distinct names; we expose helpers
# below to keep the per-Spot uuids in lockstep with the rest of the codebase.
def head_sensor_uuids(spot_index: int) -> Tuple[str, str]:
    """Return ``(rgb_uuid, depth_uuid)`` for Spot ``spot_index``."""
    return (f"spot_{spot_index}_head_rgb", f"spot_{spot_index}_head_depth")


def _head_sensor_specs(
    spot_index: int,
) -> Tuple[Tuple[str, "habitat_sim.SensorType"], ...]:
    rgb_uuid, depth_uuid = head_sensor_uuids(spot_index)
    return (
        (rgb_uuid, habitat_sim.SensorType.COLOR),
        (depth_uuid, habitat_sim.SensorType.DEPTH),
    )


def _make_camera_agent(
    sensor_uuids: Iterable[Tuple[str, "habitat_sim.SensorType"]],
    image_hw: Sequence[int],
    hfov_deg: float,
) -> habitat_sim.agent.AgentConfiguration:
    """Build an AgentConfiguration whose body holds N camera sensors.

    All sensors share the same body node; per-sensor local rotation is set at
    runtime by ``PanTiltHead`` for tilt control.
    """
    cfg = habitat_sim.agent.AgentConfiguration()
    specs = []
    for uuid, kind in sensor_uuids:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = kind
        spec.resolution = list(image_hw)
        spec.hfov = float(hfov_deg)
        # Local position is zero - we drive the parent agent's transform from PanTiltHead.
        spec.position = [0.0, 0.0, 0.0]
        specs.append(spec)
    cfg.sensor_specifications = specs
    return cfg


def make_sim(
    scene_id: str,
    scene_dataset_config_file: str,
    image_hw: Sequence[int] = (720, 1280),
    spot_head_hfov_deg: float = 110.0,
    observer_hfov_deg: float = 90.0,
    enable_physics: bool = True,
) -> habitat_sim.Simulator:
    """Construct a habitat-sim Simulator with the M1 multi-agent camera rig.

    Args:
        scene_id: Path to the ``.scene_instance.json`` (or .glb fallback) to load.
        scene_dataset_config_file: Path to the dataset's ``.scene_dataset_config.json``.
        image_hw: (height, width) of every camera sensor.
        enable_physics: Required (True) for articulated-object support.

    Returns:
        Initialised habitat_sim.Simulator with three agents.
    """
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_id
    sim_cfg.scene_dataset_config_file = scene_dataset_config_file
    sim_cfg.enable_physics = enable_physics

    observer_cfg = _make_camera_agent(
        [("observer_rgb", habitat_sim.SensorType.COLOR)],
        image_hw,
        hfov_deg=observer_hfov_deg,
    )
    spot_0_head_cfg = _make_camera_agent(
        _head_sensor_specs(0), image_hw, hfov_deg=spot_head_hfov_deg
    )
    spot_1_head_cfg = _make_camera_agent(
        _head_sensor_specs(1), image_hw, hfov_deg=spot_head_hfov_deg
    )

    return habitat_sim.Simulator(
        habitat_sim.Configuration(
            sim_cfg, [observer_cfg, spot_0_head_cfg, spot_1_head_cfg]
        )
    )


# Stable agent indices used elsewhere in the project.
OBSERVER_AGENT_ID: int = 0
SPOT_0_HEAD_AGENT_ID: int = 1
SPOT_1_HEAD_AGENT_ID: int = 2
