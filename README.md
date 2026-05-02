# mumt

Habitat 3 simulation with 2 Spot-like robots and 1 humanoid in an HSSD scene.
Eventual VR embodiment of the human via Habitat-HITL.

## Status

- **M1 (done):** isolated environment, fresh habitat-sim + habitat-lab source clones,
  one HSSD scene + Spot URDF + humanoid URDFs, headless render of 2 kinematic Spots
  and 1 kinematic humanoid with per-Spot pan/tilt RGB-D heads.
- **M2a (in progress):** keyboard teleop of Spot 0 with continuous diff-drive
  velocity + arrow-key pan/tilt, live split-screen display.
- **M2b/c (planned):** autonomous geodesic-follower Spots, SMPL-X walking humanoid.
- **M3 (planned):** HITL desktop viewer -> HITL VR with USB-tethered Quest -> floating
  HUD panels (Spot feeds + custom map).

See `/home/vignesh/.cursor/plans/` for the live planning docs and the
[Milestones](#milestones) section below for the running technical journal.

## Layout

```
mumt/
  .venv/                              # Python 3.10 venv (isolated)
  third_party/
    habitat-sim/                      # fresh clone, built from source
    habitat-lab/                      # fresh clone, pip -e installed
  data/                               # all assets local to this repo
    scene_datasets/hssd-hab/          # one HSSD scene (gated)
    robots/hab_spot_arm/              # Spot URDF + meshes
    humanoids/humanoid_data/          # SMPL-X humanoid URDFs
  mumt_sim/
    scene.py     agents.py     spawn.py     pan_tilt.py
    teleop.py    display.py
  scripts/
    01_install_env.sh                 # bootstrap venv + clone + build
    02_fetch_assets.sh                # hf login + dataset_downloader + hf download
    render_two_spots_one_human.py     # M1 deliverable
    teleop_spot.py                    # M2a deliverable
  renders/                            # output PNGs (gitignored)
```

## Setup

Run once, in order:

```bash
bash scripts/01_install_env.sh
bash scripts/02_fetch_assets.sh
```

Then activate the venv and run the M1 sanity render:

```bash
source .venv/bin/activate
python scripts/render_two_spots_one_human.py
```

Outputs land in `renders/`.

## Prerequisites

- Linux with X / OpenGL.
- Python 3.10 (we use the system `/usr/bin/python3.10`).
- Build tools: `cmake`, `ninja`, `git`, `gcc`/`g++`.
- CUDA toolkit if building habitat-sim with `--with-cuda` (optional; CPU-only build also works).
- A Hugging Face account with access granted to `hssd/hssd-hab` (gated dataset).

## Milestones

This is the running technical journal. Each milestone documents not just what was
built but the conventions it established and the problems it solved, so future
milestones can grep for prior art instead of re-discovering gotchas.

Each entry uses the same five-block schema:

1. **Goal** - one sentence.
2. **Deliverable / how to run** - the script(s), the command, the outputs.
3. **Conventions established here** - new project-wide invariants. Future
   milestones treat these as load-bearing.
4. **Problems hit and how we solved them** - "Problem -> Fix" pairs, each with
   the file path where the fix lives.
5. **What we deliberately punted** - scoped-out items future milestones will revisit.

### M1 - Static group shot (2 Spots + 1 humanoid in HSSD)

**Goal:** load HSSD scene `102344049`, place two kinematic Spots and one humanoid
in a 1 m equilateral triangle facing inward, render observer + per-Spot head cams.

**Deliverable / how to run:**

```bash
bash scripts/01_install_env.sh        # one-time
bash scripts/02_fetch_assets.sh       # one-time
source .venv/bin/activate
python scripts/render_two_spots_one_human.py
# -> renders/{observer, spot_{0,1}_head_rgb, spot_{0,1}_head_depth}.png
```

**Conventions established here:**

- Project layout: `third_party/{habitat-sim, habitat-lab}` are submodules-by-clone,
  `data/` holds gated assets, `.venv` at repo root, no global Python state.
- `yaw=0` means "body forward along world `+X`" for both Spot and humanoid.
  `add_kinematic_humanoid` silently adds `+pi/2` to the URDF rotation to honour
  this (`mumt_sim/agents.py: _HUMAN_FORWARD_YAW_OFFSET`).
- Each Spot's head sensors get unique uuids (`spot_{i}_head_rgb` /
  `spot_{i}_head_depth`) via `mumt_sim.scene.head_sensor_uuids` - habitat-sim's
  sensor registry is keyed globally by uuid, so duplicates collide silently.
- The habitat-sim camera's local `-Z` is mapped onto the Spot URDF's local `+X`
  by `PanTiltHead._BODY_TO_CAMERA_BASIS` so the camera looks where the snout points.
- `AgentState.rotation` must be a 4-tuple `[x, y, z, w]`, not a `magnum.Quaternion`.
  Always unpack: `[q.vector.x, q.vector.y, q.vector.z, q.scalar]`.
- HSSD scenes ship without a navmesh; recompute on load with
  `include_static_objects = True` so cars/furniture become obstacles.
- Default Spot head hfov is 110 deg (`mumt_sim/scene.make_sim`) so adjacent agents
  in a 1 m triangle stay in frame.

**Problems hit and how we solved them:**

- `humanoid_data` UID gone from `datasets_download`
  -> renamed to `habitat_humanoids` in `scripts/02_fetch_assets.sh`.
- HSSD download blew up against HF rate limits when grabbing all of `objects/**`
  -> parse the chosen `scene_instance.json`, ask `HfApi.list_repo_files` for the
     ~256 files actually referenced by it, `xargs` them to one `hf download` call.
- `hf download --include` only accepts one pattern per flag
  -> one `--include` flag per pattern in `scripts/02_fetch_assets.sh`.
- Both Spot head cams produced byte-identical frames
  -> sensor uuid collision; per-agent uuids in `mumt_sim/scene.head_sensor_uuids`.
- `get_sensor_observations()` returned only agent 0's frame
  -> pass an explicit list `[OBSERVER, SPOT_0_HEAD, SPOT_1_HEAD]`.
- `AgentState.set_state` crashed with `TypeError` on a `magnum.Quaternion`
  -> unpack to `[x, y, z, w]`; `mumt_sim/pan_tilt.PanTiltHead.sync`.
- Each Spot's camera pointed 90 deg to its right
  -> `_BODY_TO_CAMERA_BASIS = R_y(-pi/2)` so camera local `-Z` aligns with body `+X`.
- HSSD `pathfinder.is_loaded` was False
  -> `recompute_navmesh` in `scripts/render_two_spots_one_human.py`.
- `find_open_spawn_spot` picked the garage; cars were not in the navmesh
  -> `include_static_objects = True` before `sim.recompute_navmesh`.
- Humanoid showed profile instead of facing the centroid
  -> SMPL-X URDF natural forward is `+Z`; `agents.py` adds `pi/2` yaw offset.

**What we deliberately punted:**

- No physics, no leg gait, no humanoid animation - all bodies are
  `MotionType.KINEMATIC` with `fixed_base=True`.
- No real-time interaction; M1 is one-shot static PNGs.
- HF token paths still rely on `hf auth login` having been run (no programmatic
  fallback). Fine for solo dev, revisit if we ever need CI.

### M2a - Keyboard teleop of Spot 0

**Goal:** drive Spot 0 around the HSSD scene from the keyboard with continuous
diff-drive velocity + arrow-key pan/tilt, while Spot 1 and the human stay frozen
in their M1 triangle. Live split-screen pygame window.

**Deliverable / how to run:**

```bash
source .venv/bin/activate
python scripts/teleop_spot.py
```

Opens an 1280x480 window: Spot 0's head RGB on the left, observer bird's-eye on
the right, with a small HUD showing position, yaw, pan, tilt, fps. Controls:

| Key            | Action                                                |
| -------------- | ----------------------------------------------------- |
| `W` / `S`      | forward / backward (default 0.8 m/s along body +X)    |
| `A` / `D`      | yaw left / right (default 90 deg/s around world +Y)   |
| `Up` / `Down`  | tilt head up / down (clamped to plus-or-minus 60 deg) |
| `Left`/`Right` | pan head left / right (free, wraps)                   |
| `Shift`        | 2x speed boost (linear + yaw) while held              |
| `R`            | reset Spot 0 to start pose, zero pan/tilt             |
| `Esc` / close  | quit                                                  |

**Conventions established here:**

- Teleop modules know nothing about pygame. Drivers fill a
  ``mumt_sim.teleop.TeleopInput`` per frame and call ``SpotTeleop.step(dt, controls)``.
  Same pattern will let an OpenXR / VR controller driver feed the same teleop in M3.
- Live render resolution (480x640 per panel) is a separate concern from the
  offline render resolution (M1 still uses 720x1280). Pass ``image_hw`` to
  ``mumt_sim.scene.make_sim`` per use case.
- Body XZ motion gets clamped against walls via ``pathfinder.try_step`` (sliding
  contact). Yaw, pan, and tilt are unconstrained except for tilt's hard plus-or-minus
  60 deg clamp inside ``mumt_sim/teleop.py``.

**Problems hit and how we solved them:**

- ``Quaternion.angle()`` always returns [0, pi] so reading initial yaw from the
  Spot AO loses sign
  -> reconstruct sign from ``rotation.axis().y`` in ``SpotTeleop.__init__``.
- pygame-side numpy arrays must be transposed: habitat hands us ``(H, W, 3)``,
  pygame expects ``(W, H, 3)``
  -> handled inside ``SplitScreenWindow._to_surface``.
- Stale ``timeout`` invocations during smoke-testing didn't kill pygame loops
  cleanly
  -> not a code issue, just confirmed clean exit on ``Esc`` / window close in
     ``scripts/teleop_spot.py``'s ``finally`` block (``window.close``,
     ``sim.close``).

**What we deliberately punted:**

- No collision avoidance vs. Spot 1 / human; both were spawned *after* the
  navmesh was baked, and they're ``KINEMATIC`` so a re-bake with
  ``include_static_objects=True`` wouldn't include them either. Driving Spot 0
  through them is allowed for now; M2b will need either ``MotionType.STATIC``
  before bake or a per-step proximity check.
- No leg gait. Spot's body slides smoothly across the floor; legs stay in
  standing pose. (Real gait is M2c-or-later, via habitat-lab's ``SpotRobot``.)
- No screenshot / recording key. Add later if needed (5 lines via ``imageio``).
- No mouse capture / FPS-style look. The arrow keys are the head; revisit when
  we add a desktop "embody the human" mode in M2b/c.
