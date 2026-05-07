# mumt

Habitat 3 simulation with 2 LLM-driven Spot-like robots and a VR-embodied
human operator in an HSSD scene, served to a Quest 2 over Habitat-HITL.

## Status

- **M1 (done):** isolated environment, fresh habitat-sim + habitat-lab source clones,
  one HSSD scene + Spot URDF + humanoid URDFs, headless render of 2 kinematic Spots
  and 1 kinematic humanoid with per-Spot pan/tilt RGB-D heads.
- **M2a (done):** keyboard teleop of Spot 0 with continuous diff-drive
  velocity + arrow-key pan/tilt, live split-screen display.
- **M2b/c (superseded):** original plan was geodesic-follower Spots + SMPL-X
  walking humanoid. We pivoted to the LLM-autonomy stack (M-Agent.\*) instead;
  legged gait + animated humanoid stay open as future polish.
- **M3a (done):** habitat-hitl runtime stack validation - Unity Editor
  on Linux acts as the VR client over localhost. No headset yet.
- **M3b (done):** Quest 2 sideload of the same Unity client. HSSD scene
  102344049 streams from a desktop habitat-hitl server over USB-tether and
  renders inside the headset, head-tracked.
- **M-Agent.2 (done):** two-spot keyboard teleop with shared top-down
  coverage map + 1 m chess-named sectors, ambient Gemini Flash Lite
  captions, per-run perception-memory JSONL, and the
  `goto`/`move`/`search`/`find`/`recall` action primitives.
- **M-Agent.3 (done):** per-Spot event-driven ReAct `AgentLoop` on top of
  those primitives, plus an LLM orchestrator that fans terminal chat to
  one or both Spots concurrently.
- **M3c (done):** custom `mumt_hitl_app` with VR-embodied human (head /
  yaw / left-stick locomotion + `teleportAvatarBasePosition` keyframes),
  right-stick teleop of Spot 0, server-side virtual-display framework,
  and a phase-E graft of the M-Agent.\* autonomy stack into VR (per-Spot
  head sensors, manual override, push-to-talk + STT, controller-pointing).

See `/home/vignesh/.cursor/plans/` for the live planning docs -- the
relevant ones since M3b are `m_agent_2_coverage_memory_teleop_*`,
`action_primitives_goto_and_move_*`, `search-sector-primitive_*`,
`recall_memory_tool_*`, `agent_loop_module_*`, and
`m3c_autonomy_integration_*`. The [Milestones](#milestones) section
below is the running technical journal.

## Layout

```
mumt/
  .venv/                              # Python 3.10 venv (isolated)
  .venv-magnum/                       # Python 3.11 venv for magnum-tools
  third_party/
    habitat-sim/                      # fresh clone, built from source
    habitat-lab/                      # fresh clone, pip -e installed
    siro_hitl_unity_client/           # Unity client + APK build dir (M3a/b)
    magnum-tools/                     # mosra/magnum-ci artifact (M3b)
  data/                               # all assets local to this repo
    scene_datasets/hssd-hab/          # one HSSD scene (gated)
    robots/hab_spot_arm/              # Spot URDF + meshes
    humanoids/humanoid_data/          # SMPL-X humanoid URDFs
  mumt_sim/
    scene.py     agents.py     spawn.py     pan_tilt.py
    teleop.py    display.py    vr_displays.py
    agent/                            # M-Agent.* autonomy stack
      coverage.py    memory.py        # 2-tier grid + perception-memory
      perception.py  detection.py     # Gemini caption / YOLOE detect clients
      recall.py      visibility.py    # text recall + greedy set-cover planner
      tools.py       loop.py          # action primitives + per-Spot ReAct agent
      orchestrator.py                 # LLM router across both Spots
      head_cam.py    voice.py         # AO-mounted RGB+depth, PTT + STT
      pointing.py                     # right-controller raycast for VR
  scripts/
    01_install_env.sh                 # bootstrap venv + clone + build
    02_fetch_assets.sh                # hf login + dataset_downloader + hf download
    03_install_unity_hub.sh           # M3a.1: apt install unityhub (sudo)
    04_install_vr_assets.sh           # M3a.2: clone unity client + magnum env
    05_process_unity_data.sh          # M3b.1: bake HSSD scene into Unity GLBs
    06_run_sim_viewer_server.sh       # M3a.3: run habitat-hitl sim_viewer on :8888
    07_run_mumt_hitl_server.sh        # M3c: run our custom mumt_hitl_app
    process_unity_hssd.py             # M3b.1: dataset processing config
    config/mumt_hitl.yaml             # M3c: Hydra config for mumt_hitl_app
    render_two_spots_one_human.py     # M1 deliverable
    teleop_spot.py                    # M2a deliverable
    teleop_two_spots_with_coverage.py # M-Agent.2 deliverable
    agent_chat_single_spot.py         # M-Agent.3 slice A deliverable
    agent_chat_multi_spot.py          # M-Agent.3 slice B deliverable
    mumt_hitl_app.py                  # M3c deliverable (the HITL server)
    find_centering_smoketest.py       # M-Agent.2 slice G dev harness
  renders/                            # output PNGs (gitignored)
  outputs/                            # per-run perception-memory JSONL + thumbs
  _data_processing_output/            # baked Unity GLBs (M3b)
```

## Setup

Run once, in order:

```bash
bash scripts/01_install_env.sh
bash scripts/02_fetch_assets.sh
```

Then activate the venv and run any of the entry points below:

```bash
source .venv/bin/activate

# M1: one-shot static render of 2 Spots + 1 humanoid (no env vars needed).
python scripts/render_two_spots_one_human.py
#   -> renders/{observer, spot_{0,1}_head_rgb, spot_{0,1}_head_depth}.png

# M2a: live keyboard teleop of Spot 0 (no env vars needed).
python scripts/teleop_spot.py

# M-Agent.2: 2-spot teleop + coverage + autonomy primitives.
#   GEMINI_API_KEY  -> required for ambient captions, search, recall
#   MUMT_YOLOE_URL  -> required for find(label); default http://localhost:8080
python scripts/teleop_two_spots_with_coverage.py

# M-Agent.3: terminal chat against one (or two) LLM-driven Spots.
python scripts/agent_chat_single_spot.py
python scripts/agent_chat_multi_spot.py

# M3c: VR HITL server for the Quest client.
#   Pure VR demo path needs no env vars; flip mumt.autonomy.enabled=true
#   in scripts/config/mumt_hitl.yaml (or override on the CLI) to bring
#   the M-Agent.* stack into VR -- then GEMINI_API_KEY / MUMT_YOLOE_URL
#   matter again, plus a host mic for push-to-talk.
bash scripts/07_run_mumt_hitl_server.sh
```

Outputs land in `renders/` (M1) and `outputs/` (M-Agent.\* memory JSONL + thumbs).

## Prerequisites

- Linux with X / OpenGL.
- Python 3.10 (we use the system `/usr/bin/python3.10`) for the main venv,
  Python 3.11 for `.venv-magnum` (M3b only; see M3b notes).
- Build tools: `cmake`, `ninja`, `git`, `gcc`/`g++`.
- CUDA toolkit if building habitat-sim with `--with-cuda` (optional; CPU-only build also works).
- A Hugging Face account with access granted to `hssd/hssd-hab` (gated dataset).
- `GEMINI_API_KEY` from Google AI Studio for the autonomy stack (ambient
  captions, `search`, `recall`, ReAct agent, orchestrator, default Gemini
  STT). Optional unless `mumt.autonomy.enabled=true` or you launch one of
  the `agent_chat_*` / `teleop_two_spots_with_coverage` scripts.
- Optional Jetson YOLOE server reachable at `MUMT_YOLOE_URL` (defaults
  to `http://localhost:8080`) for the `find(label)` primitive. The rest
  of the stack runs without it -- `find()` just surfaces a per-call error.
- Host microphone reachable through `sounddevice` for the M3c
  push-to-talk path. `ELEVENLABS_API_KEY` is only needed if you switch
  `mumt.autonomy.stt.backend` from `gemini` to `elevenlabs`.
- `adb` (`android-tools-adb`) for the Quest sideload + reverse-tunnel
  workflow in M3b / M3c.

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

Opens a 1280x480 OpenCV window: Spot 0's head RGB on the left, observer bird's-eye
on the right, with a small HUD showing position, yaw, pan, tilt, fps. Click the
window once after it appears so it has keyboard focus. Controls:

| Key            | Action                                                |
| -------------- | ----------------------------------------------------- |
| `W` / `S`      | forward / backward (default 0.8 m/s along body +X)    |
| `A` / `D`      | yaw left / right (default 90 deg/s around world +Y)   |
| `Up` / `Down`  | tilt head up / down (clamped to plus-or-minus 60 deg) |
| `Left`/`Right` | pan head left / right (free, wraps)                   |
| `Shift`        | 2x speed boost (linear + yaw) while held              |
| `R`            | reset Spot 0 to start pose, zero pan/tilt             |
| `Esc` / close  | quit                                                  |

Note: keys are captured globally via ``pynput`` for accurate held-key state,
so the cv2 window does NOT need keyboard focus, but typing in another app
(Slack, terminal, editor) will also drive Spot. Quit before doing that.

**Conventions established here:**

- Teleop modules are framework-agnostic. Drivers fill a
  ``mumt_sim.teleop.TeleopInput`` per frame and call ``SpotTeleop.step(dt, controls)``.
  The window emits a ``mumt_sim.display.InputState`` (semantic axis flags) and
  the script translates between the two. Same pattern will let an OpenXR / VR
  controller driver drop in for M3 without touching ``teleop.py``.
- Live render resolution (480x640 per panel) is a separate concern from the
  offline render resolution (M1 still uses 720x1280). Pass ``image_hw`` to
  ``mumt_sim.scene.make_sim`` per use case.
- Body XZ motion gets clamped against walls via ``pathfinder.try_step`` (sliding
  contact). Yaw, pan, and tilt are unconstrained except for tilt's hard plus-or-minus
  60 deg clamp inside ``mumt_sim/teleop.py``.
- Live windowing uses **OpenCV** for display + **pynput** for input (see
  problems below). pynput hooks real X11 KeyPress/KeyRelease events in a
  background thread, so 'is W held' is precise instead of an auto-repeat guess.

**Problems hit and how we solved them:**

- Pygame's window segfaulted habitat-sim's renderer on the first
  ``get_sensor_observations`` after ``pygame.display.set_mode`` (NVIDIA GLX
  context conflict; visible as ``nv-implementation-color-read-format-dsa-broken``
  in Magnum's workaround log)
  -> dropped pygame entirely; ``mumt_sim/display.py`` now uses
     ``cv2.imshow`` + ``cv2.waitKeyEx``. cv2's GTK/Qt window has no GL state of
     its own so habitat-sim keeps its context.
- cv2 has no native held-key state (only one-shot ``waitKey`` events). First
  attempt: emulate via a 120 ms last-seen window over keycodes. Felt visibly
  laggy because Linux's keyboard auto-repeat doesn't start until ~250 ms after
  press, so every fresh keystroke had a dead-zone between the initial press
  and the first auto-repeat
  -> swapped to ``pynput.keyboard.Listener`` running in a daemon thread; it
     reports real KeyPress / KeyRelease events at the X11 layer, no
     auto-repeat dependency. Implementation in ``mumt_sim/display._PynputTracker``.
     Caveat: pynput captures globally, so don't type elsewhere while teleop
     is open. Wayland-only (no Xwayland) sessions will see no keys.
- ``Quaternion.angle()`` always returns [0, pi] so reading initial yaw from the
  Spot AO loses sign
  -> reconstruct sign from ``rotation.axis().y`` in ``SpotTeleop.__init__``.
- habitat hands us ``(H, W, RGBA)`` uint8, cv2 wants BGR
  -> ``SplitScreenWindow._to_bgr`` strips alpha and runs ``COLOR_RGB2BGR``.

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

### M3a - Habitat-HITL runtime stack validation

**Goal:** prove that habitat-hitl's official VR pipeline (``pick_throw_vr``
example + Unity client) runs end-to-end on this Linux box, with Unity Editor
acting as the client over localhost. No headset yet; this checkpoint exists
so we can fail fast on Unity / dataset-conversion / networking issues before
investing in headset deployment (M3b) or a custom mumt VR app (M3c).

**Architecture refresher:** habitat-hitl's VR is *not* Quest Link. The Linux
process is a server that streams scene state over TCP to a Unity-based client.
The client renders locally on whatever hardware it runs on (Unity Editor on
Linux for M3a, sideloaded Android APK on the Quest for M3b). USB tether to the
Quest is just TCP-over-USB; Quest Link / Air Link are not involved.

**M3a.1 - Install Unity Hub + project-specific Editor:**

```bash
sudo bash scripts/03_install_unity_hub.sh        # apt repo install
git clone --depth 1 https://github.com/eundersander/siro_hitl_unity_client.git \
    third_party/siro_hitl_unity_client            # already cloned by 04_install_vr_assets.sh
unityhub                                          # launch the GUI
```

Then in the Unity Hub GUI (manual, one-time):

1. Sign in with a Unity account (Personal license is free; create one at
   id.unity.com if you don't have one).
2. Click ``Open`` -> ``Add project from disk``, pick
   ``third_party/siro_hitl_unity_client``. Hub will say
   ``Editor 2022.3.7f1 is required, install it?`` -> yes. This is the exact
   patch the project targets; opening the project directly avoids the
   guess-the-LTS-patch problem.
3. On the install modules page, tick **Android Build Support** (we'll need it
   for M3b; cheaper to add now than re-trigger the editor download later) and
   accept its sub-modules (``OpenJDK``, ``Android SDK & NDK Tools``).
4. Wait for the editor + modules to download. Disk: ~12 GB. Goes under
   ``~/Unity/Hub/Editor/2022.3.7f1/`` by default; outside this repo on purpose.

Validation: Unity Hub's ``Installs`` tab shows ``2022.3.7f1`` with the
Android module checkmark, no red errors.

**M3a.2 - Local Unity sanity test (no server, no data processing):**

The Unity client ships with one bundled test asset, so we can validate the
Unity-side install in isolation before fighting with the dataset pipeline.

1. In Unity Hub, double-click the ``siro_hitl_unity_client`` project to open
   it in Unity Editor 2022.3.7f1.
2. ``File`` -> ``Open Scene`` -> ``Assets/Scenes/PlayerVR``.
3. From the Project pane, navigate to ``Assets/temp/``, drag
   ``106879104_174887235`` into the Scene pane. You should see HSSD walls + floors.
4. Hit Play (top-center button). After a brief load, you should see a
   first-person view with simulated VR controllers, navigable via WASD + mouse.

Validation: Play mode shows a stage and the XR Device Simulator overlay; no
red errors in the Console pane. After validating, drag the dropped object out
of the Hierarchy / undo it so the scene goes back to clean state.

**M3a.3 - Localhost server smoke test:**

```bash
# Terminal 1: habitat-hitl server, headless, on ws://127.0.0.1:8888
bash scripts/06_run_sim_viewer_server.sh
# wait for: "NetworkManager started on networking thread.
#            Listening for client websocket connections on port 8888..."
```

```
Terminal 2: Unity Editor still has Assets/Scenes/PlayerVR open from M3a.2.
            Hit ▶ Play. The Console should show "Connected to ws://127.0.0.1:8888"
            and a recurring "Message rate: ~10/s" line. The Game view streams
            HSSD scene 102344049 from habitat instead of the bundled test asset.
```

Validation: server log prints ``Client is ready!`` exactly once per Unity Play
session; Unity Console has no red errors and shows steady message rate; Game
view shows the streamed scene.

To stop: Ctrl-C the server, then click ▶ in Unity to exit Play.

**Conventions established here:**

- Linux + Wired Quest = **Unity APK + TCP-over-USB**, never Quest Link / Air
  Link. (Linux has no first-party Quest Link support; the apt-installed
  ``unityhub`` is the official path.)
- Unity Editor lives outside this repo (``~/Unity/Hub/``) - it's a
  multi-project tool, not a per-project dependency.
- ``scripts/06_run_sim_viewer_server.sh`` is the canonical localhost server.
  It runs habitat-hitl's ``sim_viewer`` example app in headless mode
  (``habitat_hitl.experimental.headless.do_headless=True`` +
  ``habitat_hitl.window=null``) so we can iterate on the server without
  fighting an extra GLX window. The Unity client is the *only* renderer at
  M3a.3; if you want to see the scene you need Unity in Play mode.
- Server port is ``8888`` because the Unity client (``NetworkClient.cs``,
  ``defaultServerPort = 8888``) is hard-coded to that. We override habitat-hitl's
  default ``18000`` instead of patching Unity, since Unity rebuilds are slow.

**Problems hit and how we solved them:**

- Originally planned to install Unity Hub via AppImage; Unity now serves Linux
  only via apt
  -> ``scripts/03_install_unity_hub.sh`` adds the Unity apt key + repo and
     installs ``unityhub`` (sudo required - no clean way around it).
- Unity Hub kept offering Unity 6.x by default; the client project targets
  Unity 2022.3 LTS specifically and Unity 6's XR packages have breaking changes
  -> open the project via "Add project from disk" so Hub auto-prompts the
     correct ``2022.3.7f1`` install. Latest 2022.3 patch (``2022.3.62f3``)
     also works and is what we landed on.
- Default ``sim_viewer`` GUI mode crashed on
  ``module '_magnum.text' has no attribute 'Renderer2D'`` (text drawer / magnum
  ABI mismatch in our pinned wheels)
  -> run server in headless mode (``do_headless=True`` + ``window=null``).
     Unity is the only renderer we care about anyway.
- Unity client (``eundersander/siro_hitl_unity_client`` HEAD ``dbfa5a6``) is
  on an older protocol than our habitat-hitl: it never sends
  ``isClientReady`` and writes ``connection_params_dict`` (snake_case) instead
  of ``connectionParamsDict`` (camelCase). The server's
  ``send_connection_record_to_main_thread`` ``assert "isClientReady" in ...``
  fired silently (``AssertionError`` with empty message), surfacing as
  ``Error receiving from client: `` followed by an instant disconnect, and
  Unity went into a 4 s reconnect loop
  -> instead of forking the abandoned upstream client we inject the missing
     field via the server's own ``mock_connection_params_dict`` test hook:
     ``+habitat_hitl.networking.mock_connection_params_dict.isClientReady=true``.
     See the comment at the top of ``scripts/06_run_sim_viewer_server.sh``.
- A failing networking subprocess (port-in-use) does not bring down the
  outer ``sim_viewer`` main process; the server keeps stepping silently and
  Unity ends up connected to whatever orphan was already on ``:8888``
  -> if you see ``OSError: [Errno 98] address already in use`` in the log,
     ``pkill -f sim_viewer.py`` and start over. Future polish: bind-check
     in the wrapper before launching.

**What we deliberately punted:**

- **Asset pipeline.** The Unity client is supposed to render scenes with full
  HSSD geometry by reading prebaked GLBs from
  ``Assets/Resources/data/`` (populated by ``habitat_dataset_processing`` +
  ``magnum-tools``). We deliberately skipped that pipeline at M3a.3:
  habitat-hitl's gfx-replay stream still drives Unity's networking,
  ``GfxReplayPlayer`` still resolves load instructions, and any unresolved
  GLB simply fails to spawn (instead of crashing the run). That's enough to
  prove the *runtime* stack works. Full asset baking moves to M3b alongside
  the Quest deploy, since both need the same prerequisite (``magnum-tools``
  GitHub-Actions artifact + ``.venv-magnum``).
- ``scripts/04_install_vr_assets.sh`` already creates ``.venv-magnum`` and
  the ``habitat_dataset_processing`` install, so M3b only needs the manual
  ``magnum-tools`` artifact download + a wrapper around ``unity_dataset_processing.py``.
- No custom HITL app yet; ``sim_viewer`` is used unmodified at M3a.3 (M3c).
- No automation of the Unity Hub login or editor install - the Unity Hub GUI
  has no headless install mode that doesn't require a Unity account login.

### M3b - Quest 2 sideload + USB-tethered VR

**Goal:** sideload the same Unity client used in M3a onto a Quest 2 over USB,
have it stream the same HSSD scene 102344049 from a desktop habitat-hitl
server, render the scene head-tracked inside the headset.

**Deliverable / how to run:**

One-time prereqs (see Problems-and-fixes for what to do when each step fails):

```bash
# Headset side: enable Developer Mode through Meta Horizon mobile app after
# creating a free developer organization at developers.meta.com/horizon/manage.
# Then on Linux, install adb:
sudo apt install -y android-tools-adb

# Asset side: download the magnum-tools artifact (~24 MB zip) from
# github.com/mosra/magnum-ci/actions/workflows/magnum-tools.yml and extract:
#   third_party/magnum-tools/linux-x64/{bin,include,lib,python}/...
# Then re-run scripts/04_install_vr_assets.sh; expect "OK".
bash scripts/04_install_vr_assets.sh

# Bake HSSD scene 102344049 into Unity-friendly geometry under
# _data_processing_output/data/. Fast (~15 s) because 02_fetch_assets.sh
# already pruned downloads to only what scene 102344049 references.
bash scripts/05_process_unity_data.sh
```

In Unity Editor (manual, one-time per asset bake):

1. ``Tools > Update Data Folder...``, paste
   ``/home/vignesh/mumt/_data_processing_output/data`` as External Data Path,
   click ``Update Data Folder``. Wait for AssetDatabase reimport.
2. ``File > Build Settings > Android > Switch Platform`` (multi-minute reimport).
3. ``Player Settings > XR Plug-in Management > Android tab > tick Oculus``.
4. ``Player Settings > Player > Android tab > Other Settings``:
   * Package Name: ``com.mumt.hitlclient``
   * Minimum API Level: 29 (Quest 2 baseline)
   * Scripting Backend: IL2CPP
   * Target Architectures: ARM64 only
5. ``Build Settings > Build`` -> ``Build/mumt_hitl_client.apk`` (~39 MB).

Sideload + run (every session, after the editor work):

```bash
adb install -r third_party/siro_hitl_unity_client/Build/mumt_hitl_client.apk

# TCP-over-USB: Quest's localhost:8888 -> our Linux box's :8888
adb reverse tcp:8888 tcp:8888

# Server (leave running)
bash scripts/06_run_sim_viewer_server.sh
```

Then put the headset on, open the **Apps** library, switch the category
dropdown to **Unknown Sources**, launch the sideloaded app. The HSSD scene
streams in head-tracked.

**Conventions established here:**

- ``.venv-magnum`` is **Python 3.11**, not 3.10. The ``magnum-tools`` artifact
  from ``mosra/magnum-ci`` dropped Python 3.10 wheels in Feb 2026; the latest
  artifacts only ship ``cpython-311`` and ``cpython-312`` ``.so`` files. The
  main project venv stays on 3.10 because that is what habitat-sim was built
  against. Two coexisting venvs is the simplest fix.
- ``adb reverse tcp:8888 tcp:8888`` is the canonical bridge. The Unity client
  hard-codes ``ws://127.0.0.1:8888`` in ``NetworkClient.cs``. Inside the Quest
  ``127.0.0.1`` resolves to the headset itself, so the reverse-tunnel is
  load-bearing - **every Quest reboot or USB replug clears it** and you need
  to re-run the command.
- ``scripts/process_unity_hssd.py`` whitelists scene ``102344049`` only and
  uses the (non-articulated) ``hssd-hab.scene_dataset_config.json`` we already
  have on disk. Future scenes go through the same script with an extra
  whitelist entry.
- The Unity APK build target is **Android API 29 + ARM64-only + IL2CPP** and
  uses the ``Oculus`` XR provider (not ``OpenXR``). Picking exactly one
  provider matters; ticking both creates a runtime conflict.

**Problems hit and how we solved them:**

- magnum-tools GitHub Actions artifact won't download just by clicking;
  GitHub gates artifacts behind a logged-in account
  -> any free GitHub login works; the page silently shows only the SHA256
     instead of a download link when you're logged out. Find the latest
     successful run of ``magnum-tools.yml``, log in, click the artifact name
     (not the hash).
- The artifact extracts to ``magnum-tools-...-linux-x64/{bin,include,lib,python}``
  with no extra wrapper directory; ``scripts/04_install_vr_assets.sh`` expects
  ``third_party/magnum-tools/linux-x64/python/...``
  -> ``mv ~/Downloads/magnum-tools-*-linux-x64 third_party/magnum-tools/linux-x64``
     puts it where the script expects.
- magnum-tools ``.so`` files are ``cpython-311``/``312``, but
  ``scripts/04_install_vr_assets.sh`` originally created the venv with
  ``python3.10``. ``import magnum`` failed with ``No module named '_corrade'``
  -> hard-pinned ``PY=python3.11`` in ``scripts/04_install_vr_assets.sh`` with
     a comment explaining why. Ubuntu 22.04 already ships ``/usr/bin/python3.11``,
     no extra apt install needed.
- Unity ``Tools > Update Data Folder`` reimport produced **328
  ``ArgumentNullException: shader``** errors at first
  -> root cause: the project is on URP, the ``GLTFUtility`` package's URP
     ``.shadergraph`` files exist but are not visible to import-worker
     processes during batch import. Mitigations applied in order:
     - Set ``Edit > Preferences > Asset Pipeline > Import Worker Count %``
       from ``25`` to ``0`` (single-process import). Cut errors to ~170.
     - Force-compile the four GLTFUtility URP shadergraphs by double-clicking
       them so they recompile on the main thread, then ``Reimport`` the data
       folder. No further reduction (still ~170).
     - **Punted** the remaining 170 furniture imports for now. The
       ``stages/102344049.glb`` (the entire HSSD shell) imported correctly
       and is what gets seen in VR; objects/furniture are missing but not
       fatal. Tracked as M3b open-issue below.
- (Known-issue, not yet fixed): **head tracking yaw/pitch axis convention is
  off**. Looking in a direction and physically turning the head produce
  inconsistent scene rotation, suggesting a coordinate-frame mismatch
  between the Quest's XR pose and what ``sim_viewer`` expects (Habitat is +Y
  up and right-handed; Unity is +Y up and left-handed; the Quest's XR
  reference frame may need an extra basis change in
  ``CoordinateSystem.cs``). Triage moves to early M3c.

**What we deliberately punted:**

- The 170 missing furniture imports. The
  ``GLTFUtility/URP/Standard (Metallic)`` shadergraph being invisible to
  import workers is a real Unity 2022 / URP 14 / GLTFUtility issue; the next
  attempt will be either a one-off ``Reimport All`` after a project restart
  with import workers disabled, or an editor pre-warm script that calls
  ``Shader.Find`` + ``ShaderUtil.CompilePass`` on each GLTFUtility shadergraph
  before running ``AssetDatabase.ImportAsset`` on the GLB folder.
- Quest controllers / hand interaction. The default sideloaded client uses
  the bundled ``XR Device Simulator`` controllers, which work but aren't
  wired into our scene. M3c.
- HUD panels, embodied human, Spot teleop from VR. All M3c.
- Lighting on the streamed HSSD stage. The stage GLB has no baked lights, so
  the VR scene is dim. M3c will either ship a light pass through
  ``sim_viewer`` config or place Unity-side lights into the ``PlayerVR`` scene.

### M-Agent.2 - Coverage substrate, perception memory, action primitives

**Goal:** build the substrate the autonomy layer reads from - a shared
top-down coverage map keyed on chess-named sectors, an append-only
perception-memory table fed by ambient Gemini captions + YOLOE detections,
and the atomic action primitives (`goto` / `move` / `search` / `find` /
`recall`) - exercised end-to-end via a 2-spot keyboard-teleop demo before
any LLM is wired up.

**Deliverable / how to run:**

```bash
source .venv/bin/activate
export GEMINI_API_KEY=...                    # for captions / search / recall
export MUMT_YOLOE_URL=http://localhost:8080  # for find(label); optional
python scripts/teleop_two_spots_with_coverage.py
```

Layout: two head-RGB panes on top + a full-width top-down coverage map on
the bottom (HiDPI-friendly vertical stack, fits a 2880x1800 laptop's
right-of-screen column without overflow). Both Spots drive simultaneously
from one keyboard:

| Key             | Action                                                         |
| --------------- | -------------------------------------------------------------- |
| `W` `A` `S` `D` | Spot 0 forward / yaw-left / back / yaw-right                   |
| arrow keys      | Spot 1 forward / yaw / back                                    |
| `Shift`         | 2x speed boost (both Spots)                                    |
| `R`             | reset both Spots to spawn                                      |
| `Tab`           | switch the **active** Spot (target of the action primitives)   |
| `G`             | `goto` the **other** Spot's coarse sector (active Spot)        |
| `M` / `N`       | `move` 1 m forward / turn 90 deg on the active Spot            |
| `F`             | `search` the active Spot's current sector                      |
| `H`             | `find` a hard-coded label (currently `"human"`) in current sector |
| `Q`             | `recall` over the perception-memory JSONL                      |
| `X`             | abort the active Spot's running primitive                      |
| `Esc`           | quit + flush logs                                              |

Each run drops a `outputs/memory_<unix>.jsonl` next to the existing
captioner thumbnails.

**Conventions established here:**

- **Two-tier grid in `mumt_sim/agent/coverage.py`.** A 10 cm fine
  occupancy grid (per-Spot `last_seen_t`, navigable / non-navigable) is
  rolled up into 1 m chess-named coarse cells (`A1..G7`). Every spatial
  concept the autonomy layer touches - goal targets, memory rows, HUD
  labels, search regions - is keyed on the chess label. The single
  conversion module also owns `world_xz_for_coarse_label`,
  `coarse_label_for_world_xz`, `sector_fine_indices`, `neighbour_labels`,
  `region_navigable_mask`, and a sensor-direct
  `cam_T_world_from_sensor(sensor)` so M3c's body-mounted sensors can
  feed the same map without going through a habitat agent.
- **Per-Spot RGB+depth update path.** Each tick we back-project the
  Spot's depth pixels through its head-cam intrinsics into world XZ and
  stamp every fine cell each pixel hits; the coverage pane shades cyan
  for Spot 0, magenta for Spot 1, blends to white where they overlap,
  and fades from full-bright (just-seen) to ~30 % brightness over 5 min.
- **Append-only `MemoryTable`** (`mumt_sim/agent/memory.py`). One writer
  thread, lock-free reads, JSONL persistence under `outputs/memory_<unix>.jsonl`.
  `MemoryRow` schema: `(t_sim, spot_id, kind, sector, body_pose,
  head_pan, head_tilt, payload)` with `kind in {"caption", "detection",
  "self"}`. `default_jsonl_path()` is the canonical path helper.
- **`OnDemand*` thread-pool wrappers** for everything that calls a
  network: `OnDemandCaptioner` over `GeminiClient`, `OnDemandDetector`
  over `YoloeClient`, `OnDemandRecaller` over `RecallClient`. Each owns
  a small `ThreadPoolExecutor`, exposes `submit(...) -> Future`, and
  has drop-oldest backpressure so the main loop never blocks on Gemini
  / YOLOE. Default models for everything: `gemini-3.1-flash-lite-preview`
  (overridable via `MUMT_AGENT_MODEL` / `MUMT_CAPTION_MODEL` /
  `MUMT_RECALL_MODEL`).
- **`CaptionWorker`** (`mumt_sim/agent/perception.py`) is the ambient
  per-Spot caption loop: ~2 Hz Gemini Flash Lite calls on the latest head
  RGB, parse `parse_ambient_caption()`, post a `kind="caption"` row into
  `MemoryTable`. Search reuses the same client with the
  `SEARCH_VIEWPOINT_PROMPT` and `parse_search_caption()`.
- **`Controller` interface** in `mumt_sim/agent/tools.py`. A primitive
  is just a `step(dt, ctx) -> Optional[PrimitiveResult]` state machine;
  the same controllers run from the hotkeys above and (M-Agent.3) from
  LLM tool calls. `ControllerCtx` carries `latest_rgb`, `latest_depth`,
  pose, coverage handle, and memory handle so primitives stay framework-
  agnostic. Every primitive returns a `PrimitiveResult(status, reason,
  t_elapsed_s, final_pose, path_followed)` with `status in {"success",
  "unreachable", "blocked", "aborted", "timeout"}`.
- **`SearchSectorController`** plans viewpoints with a random-sample
  greedy set-cover in `mumt_sim/agent/visibility.py`: 100 sample
  positions x 12 yaws inside the target sector + 8-neighbour ring, FOV
  cone (default 70 deg HFOV from Spot's head sensor) and Bresenham LOS
  on the navigable mask, K=6 viewpoints or until next-best gain < 10
  cells. Each viewpoint is driven via `GotoController` ->
  `MoveController(dyaw)` to face -> synchronous Gemini search-prompt
  caption -> append `(pose, caption)` to `SearchResult`. Tour order is
  TSP-cheap (nearest-neighbour) since the optimiser bake (`eb421b0`).
- **`FindLabelController`** pipelines YOLOE detections on the latest
  head RGB across the same viewpoint tour; on first hit it switches to
  a depth-driven approach motion (head-cam depth median in the bbox ->
  goto the back-projected world XYZ minus standoff). Returns
  `FindResult(label, observations=[...], approach_pose, status)`.
- **`RecallController`** dumps every `MemoryRow` into a single Gemini
  Flash Lite prompt (text-only `RecallClient`) and returns
  `RecallResult(answer, n_rows, t_call_s)`. No embeddings, no text
  search - the dump is short enough that LLM context handles it.
- **`SpotTeleop.drive(dt, fwd_mps, lat_mps, yaw_rps)`** is the
  continuous-velocity control path the controllers use; the M2a
  `step(dt, controls)` keeps the boolean WASD path by translating into
  `drive(...)`. Lateral support exists at the API surface even though
  no current controller uses it.

**Problems hit and how we solved them:**

- **Open YOLOE on a rotating class list crushed Jetson throughput**
  (~0.7 FPS on `set_classes` reload vs. ~9 FPS on a fixed list)
  -> freeze the open-vocab class list at startup
  (`YoloeClient.open_classes` env-overridable via `YOLOE_CLASSES`),
  `OnDemandDetector` enforces a single-flight queue with drop-oldest
  backpressure so the worker stays on the fast path.
- **`cv2` window + habitat-sim renderer on the same NVIDIA GLX context
  segfaulted on the first sensor obs** (same class of crash as the M2a
  pygame issue, but now triggered by `cv2.namedWindow` after the
  habitat-sim EGL/GL context binds first) -> warm cv2 with a throwaway
  window before importing habitat-sim (the `_mumt_cv2_warmup` dance at
  the top of `scripts/teleop_two_spots_with_coverage.py`,
  `agent_chat_*.py`, etc.). Standardised across every script that opens
  a cv2 window alongside habitat-sim.
- **`find_open_spawn_spot` kept landing the humanoid inside couches.**
  HSSD couches sit at floor Y with a wide free hemisphere above the
  seat, so `distance_to_closest_obstacle` ranks them as "open spots"
  -> floor-level filter (reject candidates where the navmesh height
  jumps inside a small radius) plus randomized top-K pick in
  `mumt_sim/spawn.py:find_open_spawn_spot`. Each restart now picks a
  fresh spawn from the top 10, so a bad pick is one Ctrl-C away.
- **`SearchSectorController` blocked the main loop while waiting on
  per-viewpoint Gemini calls** (~1 s each, plus driving) -> pipeline
  the captions via `OnDemandCaptioner.submit()` so the next viewpoint
  starts driving while the previous one's caption is still in flight.
  Aggregated in the `eb421b0` slice-F+ optimisation pass alongside the
  TSP tour-order tweak and a lighter planner.

**What we deliberately punted:**

- **No LLM in this milestone.** Hotkeys drive everything; `recall` is
  the only LLM call in the loop. The full chat / ReAct integration
  lands in M-Agent.3.
- **Per-Spot pan / tilt sweeps.** Heads stay locked at the initial
  slight downtilt; the body's yaw does the sweeping. Trivial to add
  back via the M2a head-control path if a future primitive needs it.
- **SLAM / GT-replacement.** Coverage uses ground-truth habitat poses
  and frustum back-projection, not SLAM. Deferred indefinitely - this
  is research scaffolding, not a robotics paper.
- **Cross-spot memory queries.** Each Spot's `recall` only sees its own
  memory rows; cross-Spot recall is a one-line filter change once we
  have a use case.

### M-Agent.3 - Per-Spot ReAct AgentLoop + LLM orchestrator

**Goal:** put each Spot under its own LLM, talk to both via a single
terminal chat, and demonstrate concurrent autonomous behaviour
(searching / recalling / approaching) on top of the M-Agent.2 primitives
without changing the primitives themselves.

**Deliverable / how to run:**

```bash
source .venv/bin/activate
export GEMINI_API_KEY=...                    # required
export MUMT_YOLOE_URL=http://localhost:8080  # for find(label); optional

# Slice A: one Spot, terminal in / world out.
python scripts/agent_chat_single_spot.py

# Slice B: two Spots + LLM orchestrator that routes user lines.
python scripts/agent_chat_multi_spot.py
```

Each user line is delivered to the orchestrator (slice B) or directly
to the agent (slice A); the agent's `<speak>` text echoes back to the
terminal prefixed with `spot> `. Special chat commands:

| Command       | Effect                                                  |
| ------------- | ------------------------------------------------------- |
| `:abort`      | tell BOTH agents to stop their running primitive        |
| `:abort 0`    | tell only Spot 0 to stop                                |
| `:quit`       | clean shutdown (flushes memory JSONL, stops workers)    |

**Conventions established here:**

- **Per-Spot event-driven `AgentLoop`** (`mumt_sim/agent/loop.py`).
  Each Spot has its own `EventBus` (bounded `queue.Queue`, drop-oldest)
  receiving `UserMessage` / `ToolStarted` / `ToolProgress` / `ToolResult`
  / `ToolFailed` / `ToolStopped`. The agent thread blocks on
  `bus.drain(timeout)` and only calls Gemini when there's something new -
  one Gemini call per **wake**, never per event. Long primitives
  (`search`, `find`) auto-emit `ToolProgress` between viewpoints so the
  agent can react mid-flight without polling.
- **Three-channel output contract.** Every model turn produces
  `<thinking>...</thinking>`, optional `<speak>...</speak>`, and exactly
  one Gemini native function call from `{goto, move, search, find,
  recall, stop, done}`. `parse_thinking_speak()` tolerates missing
  tags. If a primitive is running and the agent emits a non-`stop`
  action, `ToolDispatcher` auto-stops the in-flight one and emits
  `ToolStopped(reason="auto: superseded")` so the agent sees what
  happened.
- **`ToolDispatcher` is the only thing the agent talks to.** It owns
  per-Spot pending / current slots and is the only path that touches
  habitat-sim - the `AgentLoop` thread never imports the sim. The main
  loop drains `try_start_pending(spot_id)` each tick (matches the
  pattern from `agent_chat_multi_spot.py` lines 556-576).
- **`OrchestratorLoop`** (`mumt_sim/agent/orchestrator.py`) is its own
  LLM with two functions: `tell(spot_id: int, text: str)` to fan a
  user message out to one or both `AgentLoop`s, and `ask_user(text)` to
  request clarification back. Kept deliberately dumb - all execution
  lives in the per-Spot agents; the orchestrator is just a router so
  failures localise to the routing decision.
- **Default models everywhere = `gemini-3.1-flash-lite-preview`.** Same
  default for agent / orchestrator / caption / recall, overridable per-
  service from `scripts/config/mumt_hitl.yaml` (M3c) or `MUMT_*_MODEL`
  env vars.
- **Trace ring buffers** (last 5 steps per agent) for HUD overlays:
  `(t_sim, thinking_oneline, speak_oneline, action_str)`. The single-
  Spot script renders these on top of the POV pane; the multi-spot
  script reuses the same panes from M-Agent.2.

**Problems hit and how we solved them:**

- **Mid-flight user message had to interrupt the agent immediately.**
  A blocking `chat.send_message()` with a Future return would have made
  the agent unresponsive between primitive ticks
  -> `EventBus` is a bounded queue and the agent thread blocks on
  `drain(timeout)`; a `UserMessage` lands on the bus the same way a
  `ToolProgress` does, so the agent wakes within `heartbeat_s` of the
  user pressing Enter. The agent then chooses whether to acknowledge
  with `<speak>`, emit `stop()`, or carry on.
- **Long primitives could starve the agent of context.** A 30 s
  `search` with no events would have looked like a hung agent
  -> controllers expose `progress_cb` (set by the dispatcher) and
  emit per-viewpoint `ToolProgress` payloads; `format_event_for_llm`
  renders them as `<event type="ToolProgress" ...>` blocks the agent
  consumes. Dispatcher caps: `max_steps=20` LLM calls per goal,
  `overall_timeout_s=600`, `per_call_timeout_s=20`.
- **Chat history grew unbounded** -> 40-turn FIFO drops the oldest
  user / model pairs while keeping the system prompt + few-shot
  examples in place.

**What we deliberately punted:**

- **Cross-Spot tool calls.** No `ask(other_spot, ...)`. The current
  workaround is the orchestrator: `tell(0, "ask spot 1 ...")`. Easy add
  later once we have a real use case.
- **Heartbeat events.** The agent only wakes on real tool / user
  events; periodic world-state ticks are a flag in `loop.py` but
  default off.
- **Vision-to-agent.** Agents only see text; ambient captions remain
  mediated through `recall`.
- **Per-Spot state in `<state>` block** is a one-line summary today
  (`coverage: 18/47 cells seen`); a richer payload waits until a
  planning workload demands it.

### M3c - VR-embodied human + LLM-driven Spots in the HITL app

Landed in two phases. Phase C/D first stood up the custom HITL app
(VR-embodied user + thumbstick teleop of Spot 0 + a server-side virtual-
display framework). Phase E then grafted the M-Agent.\* autonomy stack
into that app behind a single Hydra flag, so the same binary either
plays the pure-VR demo or runs both Spots under their own LLMs.

**Goal:** a single HITL server (`scripts/mumt_hitl_app.py`) where the
user wears a Quest, *is* the humanoid, drives Spot 0 from the right
thumbstick, sees Spot 1 doing its own thing under LLM control, talks
to the orchestrator via push-to-talk, and points at things in the
scene to disambiguate references.

**Deliverable / how to run:**

One-time prereq (replaces the `06_run_sim_viewer_server.sh` flow from
M3a/b - same APK, same `adb reverse tcp:8888 tcp:8888`):

```bash
adb install -r third_party/siro_hitl_unity_client/Build/mumt_hitl_client.apk
adb reverse tcp:8888 tcp:8888
```

Server (every session):

```bash
# Pure VR demo (no LLM, no captions, no detection).
bash scripts/07_run_mumt_hitl_server.sh

# VR + autonomy (per-Spot LLMs + orchestrator + STT + pointing).
GEMINI_API_KEY=... \
MUMT_YOLOE_URL=http://localhost:8080 \
bash scripts/07_run_mumt_hitl_server.sh \
    mumt.autonomy.enabled=true mumt.autonomy.stt.enabled=true
```

Quest control mapping (no APK rebuild needed - all bindings are
server-side state machines reading `ClientState`):

| Quest input              | Effect                                                                  |
| ------------------------ | ----------------------------------------------------------------------- |
| Headset pose             | Drives the humanoid's pelvis XZ + horizontal yaw                        |
| Left thumbstick          | Walks the humanoid; new target shipped as `teleportAvatarBasePosition`  |
| Right thumbstick         | Drives Spot 0 (forward + yaw, navmesh-clamped via `pathfinder.try_step`)|
| Right A (`XRButton.ONE`) | Cycles HUD: `map` -> `spot0_pov` -> `spot1_pov` -> back                 |
| Right B (`XRButton.TWO`) | **Hold** to take manual override of Spot 0 (LLM resumes 0.6 s after release) |
| Right index trigger      | **Hold** for push-to-talk + pointing raycast (release transcribes + posts) |
| Right thumbstick click   | Reset Spot 0 to spawn                                                   |

**Conventions established here:**

- **Phase C/D - VR baseline.**
  - **Custom HITL app** = `scripts/mumt_hitl_app.py` + Hydra config
    `scripts/config/mumt_hitl.yaml` + launcher
    `scripts/07_run_mumt_hitl_server.sh`. Mirrors the `sim_viewer`
    handshake conventions from M3a (headless, port 8888, mock
    `isClientReady`) so the existing Quest APK keeps working.
  - **User is the humanoid.** Per frame we snap the humanoid AO's pelvis
    to `(headset.x, _humanoid_pelvis_lift_m, headset.z)` and rotate to
    the headset's horizontal facing. Locomotion intent from the left
    thumbstick is integrated into a target XZ on the navmesh and shipped
    as a `teleportAvatarBasePosition` keyframe so the Quest's XR origin
    follows along (handled client-side by `AvatarPositionHandler.cs` in
    the existing `siro_hitl_unity_client` HEAD `dbfa5a6`).
  - **`_LiftedSpotTeleop`** subclass writes the Spot AO's translation at
    `navmesh_y + spot_base_lift_m` (~0.48 m, mirrors habitat-lab's
    `SpotRobot` base offset) so the kinematic Spot doesn't sink into
    the floor each push. Same fix shape as the humanoid pelvis lift.
  - **Server-side virtual-display framework** in `mumt_sim/vr_displays.py`.
    `DisplayManager` registers `Display` providers (each produces a
    `PIL.Image` per tick) and pushes them under the `mumtDisplays` key
    of the per-user keyframe message. The Unity client is a dumb
    canvas: it gets `create` / `update` / `setVisible` / `destroy`
    events with base64 JPEG payloads and paints the resulting quads
    against `head` / `world` / `left_hand` / `right_hand` anchors. **No
    APK rebuild for new HUDs.** Bandwidth budget: 256x192 JPEG q70 ~=
    8 kB/frame, two POV displays at 15 Hz ~= 240 kB/s over adb-reverse
    localhost.
  - **Built-in displays:** `SpotPovDisplay` (first-person camera mounted
    on a Spot AO via `sim.create_sensor(spec, spot_ao.root_scene_node)`),
    `TopDownMapDisplay` (coverage-aware top-down map), `TextDisplay`
    (callable -> string -> JPEG), `AlertWedgeDisplay` (warning panel).
    The HUD cycle is `[map, spot0_pov, spot1_pov]` (3 modes since the
    autonomy phase asked for it); status text is a permanent panel
    outside the cycle.

- **Phase E - autonomy in VR.** Everything below is gated by
  `mumt.autonomy.enabled` (default `false`). The pure-VR path stays
  pixel-identical when the flag is off, so the demo doesn't regress
  if a host is missing `google-genai` / `requests` / `sounddevice`.
  - **Per-Spot RGB+depth `SpotHeadCam`** in `mumt_sim/agent/head_cam.py`.
    Both sensors are parked on the Spot AO's root scene node (same
    pattern `SpotPovDisplay` already used) so head pose tracks the body
    automatically and the autonomy stack + the HUD POV display share a
    single render path. `cam_T_world()` reads
    `self._color_sensor.node.absolute_transformation()` directly;
    `CoverageMap.cam_T_world_from_sensor(sensor)` was added to consume
    that without going through a habitat agent.
  - **Per-tick order in `AppStateMumt.sim_update`** (load-bearing):
    1. read `RemoteClientState`,
    2. manual-override gate,
    3. step the active controller for each Spot,
    4. render the `SpotHeadCam` RGB + depth,
    5. update `CoverageMap` from depth + `cam_T_world`,
    6. `CaptionWorker.post_observation(...)`,
    7. refresh the per-Spot `state_snapshot` dict that `AgentLoop.get_state`
       reads (includes `user_pose_xz` from the humanoid AO so the
       orchestrator can refer to "the user").
  - **Manual override of Spot 0** = right B held. While held,
    `dispatcher.request_stop(0)` fires once and the right thumbstick
    is rerouted to a direct teleop of Spot 0 via `_drive_spot0_from_right_thumbstick`.
    On release we wait `_OVERRIDE_RELEASE_S = 0.6` of deadzone-stick-time
    before letting the LLM regain Spot 0; short enough to feel
    responsive, long enough to ride out brief returns to neutral.
  - **Push-to-talk + STT** in `mumt_sim/agent/voice.py`. `PushToTalkRecorder`
    opens a `sounddevice.InputStream` at 16 kHz mono on right-trigger
    rising edge and drains on falling edge. Two STT backends: Gemini
    (default; reuses `GEMINI_API_KEY`) and ElevenLabs Scribe (needs
    `ELEVENLABS_API_KEY` + the `elevenlabs` SDK). All four optional
    dependencies (`google-genai`, `requests`, `sounddevice`, `elevenlabs`)
    are import-guarded - missing any one of them surfaces a single
    startup warning and the rest of the stack keeps working.
  - **Pointing** in `mumt_sim/agent/pointing.py`. While PTT is active,
    we sample the right-controller pose every tick and run two
    raycasts: `sim.cast_ray()` against the scene and an analytic plane
    intersection against the HUD top-down quad. Closest hit wins; on
    PTT release the transcript is prepended with
    `(user pointed to (x, y, z))` before being posted to the
    orchestrator. Toggleable under `mumt.pointing.{enabled, max_dist_m}`
    (default on, 25 m).
  - **Hydra config** `scripts/config/mumt_hitl.yaml` exposes the entire
    autonomy / STT / pointing surface (`mumt.autonomy.{enabled,
    agent_model, orchestrator_model, caption_model, recall_model,
    yoloe_url, stt.{enabled, backend, model, sample_rate}}`,
    `mumt.pointing.{enabled, max_dist_m}`,
    `mumt.{spot_base_lift_m, humanoid_pelvis_lift_m}`). Defaults match
    what the M-Agent.\* scripts use so behaviour is consistent across
    entry points.
  - **Optional dependencies** are split into `[autonomy]` (`google-genai`,
    `requests`, `opencv-python`) and `[voice]` (`elevenlabs`, `sounddevice`)
    extras in `pyproject.toml`; `requirements.txt` keeps the union for
    the dev workflow. The basic VR demo doesn't pull any of them.

**Problems hit and how we solved them:**

- **`SimDriver` originally skipped texture loading**, so adding more
  `CameraSensor`s post-init returned blank rasters
  -> a small patch we maintain forces texture loading; once that
  landed `SpotPovDisplay` + `SpotHeadCam` both render correctly even
  though they're attached after the initial sim configure.
- **Three+ extra `CameraSensor`s per tick (per-Spot RGB + depth + HUD
  POV) tanked frame rate** when all four ran at the same 480x640 the
  M-Agent.2 teleop uses
  -> the autonomy stack runs head sensors at 320x240 (good enough
  for captions, detection, depth back-projection) independently of
  the HUD POV resolution. Tunable per-display in `vr_displays.py`.
- **`ClientState.avatar.hands` arrives in habitat coords, but the
  Unity-side controller forward is `-Z` Unity** (left-handed)
  -> pointing logic builds the world-space ray from the right-hand
  pose and rotates Unity's `-Z` to habitat's `+X` via the same basis
  trick `PanTiltHead` uses. Documented in `pointing.py`.
- **`sim.cast_ray` availability differs between habitat-sim builds.**
  On the HITL build it's there but the API path varies
  -> `pointing.raycast_world` probes both `Simulator.cast_ray` and
  `PathFinder.cast_ray` at startup; if neither is available we fall
  back to HUD-quad-only pointing and log a warning.
- **`_LiftedSpotTeleop._push()` is called from the base
  `__init__`** before subclass attributes are set
  -> the subclass sets `self._body_lift_y` *before* delegating to
  `super().__init__()` and re-anchors `state.position` to the
  navmesh level after, so subsequent `try_step`s stay consistent
  with the lifted AO write.
- **Autonomy state crossed thread boundaries** (per-Spot snapshots
  read by `AgentLoop`, written by the main sim thread)
  -> a single `threading.Lock` per `AppStateMumt._state_lock`
  guards a `state_snapshots: List[Dict]`; `AgentLoop.get_state`
  takes a snapshot copy under the lock and returns. No habitat-sim
  call ever happens off-thread.

**What we deliberately punted (still on the radar):**

- **Quest head-tracking yaw/pitch axis convention** from M3b is still
  off; physical head turns produce inconsistent scene rotation.
  Triage moved from "early M3c" to "after autonomy lands"; the bug
  is in `CoordinateSystem.cs` on the Unity side and needs an APK
  rebuild.
- **The 170 missing furniture imports from M3b** are still missing -
  the stage GLB renders, but the objects don't. No new info on the
  GLTFUtility URP shader-import workers issue.
- **Cross-Spot tool calls** (`ask(other_spot, ...)`). Same situation
  as M-Agent.3: the orchestrator is the workaround for now.
- **Embodied SMPL-X gait.** The humanoid root snaps to the headset
  but legs don't animate; we hide the avatar in the GUI viewport
  (`hide_humanoid_in_gui: True`) so the user doesn't see their own
  pelvis. A future multi-user observer view will need an animated
  rig.
- **Whisper / on-device STT.** Earlier plan called for `faster-whisper`
  but we landed on a Gemini STT default (no extra model download,
  no GPU) plus an ElevenLabs Scribe option for paid users. Local
  Whisper can slot back in as a third backend if API latency becomes
  a problem.
