#!/usr/bin/env python3
"""Asset bake: process HSSD scene 102344049 + Spot URDF + humanoid GLBs.

Run via scripts/05_process_unity_data.sh - that wrapper activates .venv-magnum
where magnum-tools and habitat_dataset_processing live (NOT in the main .venv).

Output goes to _data_processing_output/data/... preserving directory structure.
The Unity Editor's "Tools > Update Data Folder..." menu copies the contents of
_data_processing_output/data/ into siro_hitl_unity_client/Assets/Resources/data/.

History:
- M3b: processed the HSSD scene 102344049 only.
- M3c: extended to bake the Spot URDF (hab_spot_arm.urdf + meshesColored/*.glb)
       and the humanoid SMPL-X mannequin meshes so the Quest client can
       actually render the kinematic agents spawned by mumt_hitl_app.py.
"""

from habitat_dataset_processing import (
    AssetPipeline,
    AssetSource,
    GroupType,
    HabitatDatasetSource,
    Operation,
    ProcessingSettings,
)


if __name__ == "__main__":
    datasets: list[HabitatDatasetSource] = [
        HabitatDatasetSource(
            name="hssd-hab",
            dataset_config="scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json",
            stages=ProcessingSettings(
                operation=Operation.PROCESS,
                decimate=False,
                group=GroupType.LOCAL,
            ),
            objects=ProcessingSettings(
                operation=Operation.PROCESS,
                decimate=True,
                group=GroupType.LOCAL,
            ),
            scene_whitelist=["102344049"],
            include_orphan_assets=False,
        ),
    ]

    # Spot + humanoid: not part of any HSSD scene_instance, so we register them
    # as additional asset globs. The URDF baker resolves the URDF's link mesh
    # references; for humanoids we just bake every GLB under humanoid_data/.
    robot_settings = ProcessingSettings(
        operation=Operation.PROCESS,
        decimate=False,           # robots are already low-poly; preserve fidelity
        group=GroupType.LOCAL,
    )
    # decimate=False for humanoids: magnum-tools' decimator hangs forever on
    # SMPL-X skinned meshes (observed during M3c phase B - workers go idle
    # for 17+ minutes with no further output). The raw GLB is ~1.9 MB so
    # shipping it un-decimated is cheap and avoids the hang.
    humanoid_settings = ProcessingSettings(
        operation=Operation.PROCESS,
        decimate=False,
        group=GroupType.LOCAL,
    )
    # Paths in AssetSource.assets are relative to --input (i.e. relative to
    # the data/ folder), matching the convention used by HabitatDatasetSource.
    # We only spawn female_0 in mumt_hitl_app.py; other variants would just
    # bloat the Unity Resources/data/ tree without being referenced.
    additional_assets: list[AssetSource] = [
        AssetSource(
            name="spot",
            assets=["robots/hab_spot_arm/meshesColored/*.glb"],
            settings=robot_settings,
        ),
        AssetSource(
            name="humanoid",
            assets=["humanoids/humanoid_data/female_0/*.glb"],
            settings=humanoid_settings,
        ),
    ]

    AssetPipeline(
        datasets=datasets,
        additional_assets=additional_assets,
        output_subdir="",
    ).process()
