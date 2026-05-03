#!/usr/bin/env python3
"""M3c custom Habitat-HITL app for the mumt project.

Spawns 2 kinematic Spots + 1 kinematic humanoid in HSSD scene 102344049 and
streams the world over the standard Habitat-HITL websocket protocol so the
``siro_hitl_unity_client`` Quest 2 APK can render it head-tracked.

Joystick handling (requires the matching Unity client extension that fills
the ``xr`` block of ClientState; see Milestone M3c phase C in the README):

- The user is embodied as the humanoid: per frame we snap the humanoid's
  XZ position to the headset pose reported by the client, and rotate the
  body to match the headset's horizontal facing. Camera comes from the
  headset itself (Unity-side), so the user is "inside" the avatar.
- The left thumbstick drives the user's locomotion. Each frame we
  integrate a velocity in the head's horizontal facing direction, clamp
  to the navmesh, then push the new target as a ``teleportAvatarBasePosition``
  keyframe message so the client shifts its XR origin accordingly. The
  current siro_hitl_unity_client (HEAD = dbfa5a6) honors that message
  via Assets/Scripts/AvatarPositionHandler.cs.
- The right thumbstick drives Spot 0: y-axis -> forward/back along the
  Spot's body forward, x-axis -> yaw rate. Movement is clamped against
  the navmesh via ``pathfinder.try_step``.

Run via ``scripts/07_run_mumt_hitl_server.sh``. The corresponding hydra
config lives in ``scripts/config/mumt_hitl.yaml``.
"""
from __future__ import annotations

import math
import os
import sys
from time import monotonic
from typing import List, Sequence

import hydra
import magnum as mn
import numpy as np

import habitat_sim
from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import register_hydra_plugins
from habitat_hitl.core.key_mapping import KeyCode
from habitat_hitl.core.user_mask import Mask

# Project root (one level above this script) must be on sys.path so the
# ``mumt_sim`` package imports work without a ``pip install -e .``.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mumt_sim.agents import (  # noqa: E402
    _HUMAN_FORWARD_YAW_OFFSET,
    add_kinematic_humanoid,
    add_kinematic_spot,
)
from mumt_sim.spawn import (  # noqa: E402
    find_open_spawn_spot,
    sample_navmesh_cluster,
)


# Spot 0 teleop tuning (matches mumt_sim.teleop defaults). Mounted here so
# downstream tweaks don't drag a config-file dependency along.
_SPOT_FORWARD_SPEED: float = 0.8
_SPOT_YAW_RATE: float = math.radians(90.0)
_HUMAN_WALK_SPEED: float = 1.0  # m/s; humans walk a touch faster than Spot
_THUMBSTICK_DEADZONE: float = 0.15


def _yaw_quat(yaw_rad: float) -> mn.Quaternion:
    return mn.Quaternion.rotation(mn.Rad(yaw_rad), mn.Vector3.y_axis())


def _yaw_facing(from_xyz: Sequence[float], to_xyz: Sequence[float]) -> float:
    """Yaw in our +Y/forward=+X convention pointing ``from_xyz`` at ``to_xyz``."""
    dx = float(to_xyz[0]) - float(from_xyz[0])
    dz = float(to_xyz[2]) - float(from_xyz[2])
    # +Y rotation: forward(yaw) = (cos yaw, 0, -sin yaw); we want forward
    # to align with (dx, ?, dz), so yaw = atan2(-dz, dx).
    return math.atan2(-dz, dx)


class AppStateMumt(AppState):
    """Custom HITL app: 2 Spots + 1 VR-embodied humanoid in HSSD."""

    def __init__(self, app_service: AppService):
        self._app_service = app_service
        self._gui_input = app_service.gui_input

        # Activate the single remote user (Quest headset).
        app_service.users.activate_user(0)

        cfg = self._app_service.config.mumt
        self._dataset = cfg.dataset
        self._scene = cfg.scene
        self._app_service.reconfigure_sim(self._dataset, self._scene)

        sim = self._app_service.sim
        # HSSD has no cached navmesh; recompute including static furniture.
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.include_static_objects = True
        sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

        # Spawn the human at the most-open point in the navmesh so the user
        # has room to walk physically and via the left-stick. We then sample
        # 2 Spot spawn points near (but not on top of) the human.
        human_xyz, human_clearance = find_open_spawn_spot(
            sim.pathfinder, min_clearance=1.0, n_samples=800,
        )
        self._humanoid_spawn: List[float] = list(human_xyz)

        # Cluster Spot spawns around the human's open spot. The cluster
        # sampler also asks for n=3 because it always seeds the first point
        # at a random navmesh sample; we discard that and re-anchor the
        # remaining two points on the human's hand-picked open spot.
        cluster_pts = sample_navmesh_cluster(
            sim.pathfinder, n=3, min_sep=1.2, cluster_radius=3.0,
        )
        # cluster_pts[0] is a random anchor; cluster_pts[1:] are 1.2-3 m
        # from it. Translating them so they're 1.2-3 m from our open spot
        # is "good enough" for the demo (yes the absolute distances drift
        # slightly, but the navmesh-snap below fixes them up).
        anchor = np.asarray(cluster_pts[0])
        offset = np.asarray(human_xyz) - anchor
        self._spot0_spawn: List[float] = list(
            sim.pathfinder.snap_point(np.asarray(cluster_pts[1]) + offset)
        )
        self._spot1_spawn: List[float] = list(
            sim.pathfinder.snap_point(np.asarray(cluster_pts[2]) + offset)
        )

        spawn_pts = [
            self._humanoid_spawn, self._spot0_spawn, self._spot1_spawn,
        ]
        centroid = np.mean(np.asarray(spawn_pts), axis=0).tolist()
        humanoid_yaw = _yaw_facing(self._humanoid_spawn, centroid)
        spot0_yaw = _yaw_facing(self._spot0_spawn, centroid)
        spot1_yaw = _yaw_facing(self._spot1_spawn, centroid)
        print(
            f"[mumt-hitl] human spawn clearance = {human_clearance:.2f} m",
            flush=True,
        )

        # Spawn kinematic articulated objects (no physics, fixed_base).
        self._humanoid_ao = add_kinematic_humanoid(
            sim, self._humanoid_spawn, yaw_rad=humanoid_yaw,
        )
        self._spot0_ao = add_kinematic_spot(
            sim, self._spot0_spawn, yaw_rad=spot0_yaw,
        )
        self._spot1_ao = add_kinematic_spot(
            sim, self._spot1_spawn, yaw_rad=spot1_yaw,
        )

        # Mutable Spot 0 teleop state, driven by right thumbstick.
        self._spot0_pos: mn.Vector3 = mn.Vector3(*self._spot0_spawn)
        self._spot0_yaw_rad: float = spot0_yaw

        # Mutable user locomotion state. Driven by left thumbstick, fed back
        # to the client via ``teleportAvatarBasePosition`` keyframe messages.
        # Y is sticky to the humanoid spawn floor height; the Unity handler
        # ignores Y delta anyway.
        self._user_target_pos: mn.Vector3 = mn.Vector3(*self._humanoid_spawn)
        self._last_left_print_sec: int = 0
        # Set in on_environment_reset and cleared once we've actually told
        # the client to move; avoids stranding the user inside furniture at
        # whatever spot their Quest's tracking origin happened to be.
        self._needs_initial_teleport: bool = True
        # Initial humanoid yaw is what we fall back to if no head pose has
        # been received yet (cold connect).
        self._humanoid_yaw_rad: float = humanoid_yaw

        # Filled per-frame in sim_update.
        self._cam_transform: mn.Matrix4 = mn.Matrix4.identity_init()

        print(
            f"[mumt-hitl] spawned humanoid at {self._humanoid_spawn}, "
            f"spot0 at {self._spot0_spawn}, spot1 at {self._spot1_spawn}",
            flush=True,
        )

    def get_sim(self):
        return self._app_service.sim

    def on_environment_reset(self, episode_recorder_dict):
        client_msg = self._app_service.client_message_manager
        if client_msg is not None:
            # Notify the client that the scene mesh changed.
            client_msg.signal_scene_change(destination_mask=Mask.ALL)
            # Schedule a one-shot teleport of the user to the humanoid spawn
            # so they don't appear inside random furniture wherever the
            # Quest's tracking origin happens to be. We can't call
            # rebase_xr_headset_position() because the deployed
            # siro_hitl_unity_client (HEAD = dbfa5a6) only honors the
            # legacy teleportAvatarBasePosition key. Inject directly.
            #
            # Why not call this here unconditionally? AvatarPositionHandler
            # diff's the target against the *current* camera position, which
            # is identity until the first head pose arrives. We delay until
            # we've heard from the client; see _maybe_send_initial_teleport.
            self._needs_initial_teleport = True

    def _embody_humanoid_from_head_pose(self) -> None:
        """Snap the humanoid AO to wherever the user's headset is reporting.

        The user's Unity-side camera comes from the Quest's tracker, so by
        co-locating the humanoid's body with the headset we get free
        embodiment: the user is "inside" the avatar without needing
        first-person camera plumbing on the server.
        """
        remote = self._app_service.remote_client_state
        sim = self._app_service.sim

        head_pos, head_rot = remote.get_head_pose(0)
        if head_pos is None:
            return  # no client connected yet; leave humanoid at spawn pose

        # Snap horizontally to navmesh; humanoid feet should stay on the floor.
        target = np.asarray([head_pos[0], head_pos[1], head_pos[2]],
                            dtype=np.float32)
        snapped = np.asarray(sim.pathfinder.snap_point(target))
        if not np.all(np.isfinite(snapped)):
            return

        # Recompute yaw from head's forward direction in world XZ. The head
        # quaternion returned by RemoteClientState is already in habitat
        # agent-space (z+ forward), so we rotate the (0, 0, 1) vector instead
        # of (0, 0, -1) as we'd do for raw Unity poses.
        forward = head_rot.transform_vector(mn.Vector3(0.0, 0.0, 1.0))
        forward.y = 0.0
        if forward.length() > 1e-3:
            forward = forward.normalized()
            self._humanoid_yaw_rad = math.atan2(-forward.z, forward.x)

        self._humanoid_ao.translation = mn.Vector3(
            float(snapped[0]), float(snapped[1]), float(snapped[2])
        )
        self._humanoid_ao.rotation = _yaw_quat(
            self._humanoid_yaw_rad + _HUMAN_FORWARD_YAW_OFFSET
        )

    def _maybe_send_initial_teleport(self) -> None:
        """Once the client is reporting head poses, snap them to the spawn.

        AvatarPositionHandler.cs diffs the target against the current camera
        position to compute the delta to apply to the XR origin. So we wait
        until we've heard a real head pose at least once; otherwise the diff
        is computed against an unset / identity transform and the user
        teleports somewhere unexpected.
        """
        if not self._needs_initial_teleport:
            return
        remote = self._app_service.remote_client_state
        head_pos, _ = remote.get_head_pose(0)
        if head_pos is None:
            return
        client_msg = self._app_service.client_message_manager
        if client_msg is None:
            return
        for user_index in self._app_service.users.indices(Mask.ALL):
            msg = client_msg.get_messages()[user_index]
            msg["teleportAvatarBasePosition"] = [
                float(self._humanoid_spawn[0]),
                float(self._humanoid_spawn[1]),
                float(self._humanoid_spawn[2]),
            ]
        self._user_target_pos = mn.Vector3(*self._humanoid_spawn)
        self._needs_initial_teleport = False
        print(
            f"[mumt-hitl] teleporting user to spawn {self._humanoid_spawn}",
            flush=True,
        )

    def _drive_user_from_left_thumbstick(self, dt: float) -> None:
        """Translate the user's XR origin by the left-thumbstick velocity.

        Frame sequence:
          1. Read the left-stick (x, y) from the rich XR input.
          2. Compute a horizontal velocity in the headset's facing frame
             (so 'stick up' = 'go where I'm looking').
          3. Integrate, clamp to the navmesh.
          4. Emit ``teleportAvatarBasePosition`` so the client shifts its
             XR origin to match. The Unity-side handler diff's this against
             the current camera position and applies the delta to the
             ``xrOriginNode`` GameObject.

        No-op until both an XR thumbstick and a head pose have been
        received from the client.
        """
        if dt <= 0.0:
            return

        sim = self._app_service.sim
        remote = self._app_service.remote_client_state
        head_pos, head_rot = remote.get_head_pose(0)
        if head_pos is None:
            return

        xr = remote.get_xr_input(0)
        lx, ly = xr.left_controller.get_thumbstick()

        # Debug: print non-zero thumbstick values once per second to confirm
        # the rich xr block is reaching the server. Remove once teleop works.
        now = int(monotonic())
        if (lx != 0.0 or ly != 0.0) and now != self._last_left_print_sec:
            rx_, ry_ = xr.right_controller.get_thumbstick()
            print(
                f"[mumt-hitl] left=({lx:+.2f},{ly:+.2f}) "
                f"right=({rx_:+.2f},{ry_:+.2f})",
                flush=True,
            )
            self._last_left_print_sec = now

        if abs(lx) < _THUMBSTICK_DEADZONE:
            lx = 0.0
        if abs(ly) < _THUMBSTICK_DEADZONE:
            ly = 0.0

        if lx == 0.0 and ly == 0.0:
            # Even when idle, keep _user_target_pos in sync with the headset
            # so resuming joystick motion doesn't snap the user to a stale
            # position (e.g. after they walk physically).
            self._user_target_pos = mn.Vector3(
                float(head_pos.x), float(head_pos.y), float(head_pos.z)
            )
            return

        forward = head_rot.transform_vector(mn.Vector3(0.0, 0.0, 1.0))
        forward.y = 0.0
        if forward.length() < 1e-3:
            return  # head is staring straight up/down; skip this frame
        forward = forward.normalized()
        # Right-hand rule: right = up x forward (we want stick.x > 0 to
        # strafe to the user's right when they look down +Z agent-space).
        right = mn.math.cross(mn.Vector3(0.0, 1.0, 0.0), forward).normalized()

        # Stick y forward, stick x strafe right.
        # Empirically (M3c phase D testing) the forward axis was reversed --
        # pushing the stick forward drove the user backward -- while strafe
        # was correct. Negating ly here keeps strafe behaviour and makes
        # "push forward = walk toward what you're looking at".
        velocity = forward * (_HUMAN_WALK_SPEED * -ly) + right * (
            _HUMAN_WALK_SPEED * lx
        )

        attempted = self._user_target_pos + velocity * dt
        clamped = mn.Vector3(
            sim.pathfinder.try_step(self._user_target_pos, attempted)
        )
        self._user_target_pos = clamped

        # Keyframe message: tell the client to teleport the avatar's base
        # to this world position. Y is ignored client-side; we still pass
        # the navmesh-snapped Y so the field is well-formed.
        client_msg = self._app_service.client_message_manager
        if client_msg is not None:
            for user_index in self._app_service.users.indices(Mask.ALL):
                msg = client_msg.get_messages()[user_index]
                msg["teleportAvatarBasePosition"] = [
                    float(clamped.x),
                    float(clamped.y),
                    float(clamped.z),
                ]

    def _drive_spot0_from_right_thumbstick(self, dt: float) -> None:
        """Apply right-thumbstick (x = yaw, y = forward) to Spot 0."""
        if dt <= 0.0:
            return

        sim = self._app_service.sim
        xr = self._app_service.remote_client_state.get_xr_input(0)
        rx, ry = xr.right_controller.get_thumbstick()

        if abs(rx) < _THUMBSTICK_DEADZONE:
            rx = 0.0
        if abs(ry) < _THUMBSTICK_DEADZONE:
            ry = 0.0

        if ry != 0.0:
            speed = _SPOT_FORWARD_SPEED * ry
            cy = math.cos(self._spot0_yaw_rad)
            sy = math.sin(self._spot0_yaw_rad)
            forward_world = mn.Vector3(cy, 0.0, -sy)
            attempted = self._spot0_pos + forward_world * (speed * dt)
            self._spot0_pos = mn.Vector3(
                sim.pathfinder.try_step(self._spot0_pos, attempted)
            )

        if rx != 0.0:
            # rx > 0 (push right) -> turn right (yaw decreases in our +Y conv).
            self._spot0_yaw_rad -= _SPOT_YAW_RATE * rx * dt

        self._spot0_ao.translation = self._spot0_pos
        self._spot0_ao.rotation = _yaw_quat(self._spot0_yaw_rad)

    def sim_update(self, dt: float, post_sim_update_dict):
        if self._gui_input.get_key_down(KeyCode.ESC):
            post_sim_update_dict["application_exit"] = True

        # Order matters: locomotion mutates client message state for THIS
        # frame; embodiment reads the latest head pose (which lags the
        # locomotion by one round-trip but converges within one tick).
        self._maybe_send_initial_teleport()
        self._drive_user_from_left_thumbstick(dt)
        self._embody_humanoid_from_head_pose()
        self._drive_spot0_from_right_thumbstick(dt)

        # In VR mode the Unity client uses its own headset camera, so the
        # server-side cam_transform is cosmetic (it would only affect the
        # server's own GUI window, which we run headless). Identity is fine.
        post_sim_update_dict["cam_transform"] = self._cam_transform


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="mumt_hitl",
)
def main(config):
    hitl_main(config, lambda app_service: AppStateMumt(app_service))


if __name__ == "__main__":
    register_hydra_plugins()
    main()
