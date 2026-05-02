# mumt

Habitat 3 simulation with 2 Spot-like robots and 1 humanoid in an HSSD scene.
Eventual VR embodiment of the human via Habitat-HITL.

## Status

- **M1 (this repo, in progress):** isolated environment, fresh habitat-sim + habitat-lab
  source clones, one HSSD scene + Spot URDF + humanoid URDFs, headless render of
  2 kinematic Spots and 1 kinematic humanoid with per-Spot pan/tilt RGB-D heads.
- **M2 (planned):** kinematic motion (geodesic-follower Spots, SMPL-X walking humanoid).
- **M3 (planned):** HITL desktop viewer -> HITL VR with USB-tethered Quest -> floating
  HUD panels (Spot feeds + custom map).

See `/home/vignesh/.cursor/plans/hssd-2-spots-1-human-sim_*.plan.md` for the full plan.

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
  scripts/
    01_install_env.sh                 # bootstrap venv + clone + build
    02_fetch_assets.sh                # hf login + dataset_downloader + hf download
    render_two_spots_one_human.py     # M1 deliverable
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
