#!/usr/bin/env python3
"""M3b asset bake: process HSSD scene 102344049 into Unity-friendly geometry.

Run via scripts/05_process_unity_data.sh - that wrapper activates .venv-magnum
where magnum-tools and habitat_dataset_processing live (NOT in the main .venv).

Output goes to _data_processing_output/data/... preserving directory structure.
The Unity Editor's "Tools > Update Data Folder..." menu copies the contents of
_data_processing_output/data/ into siro_hitl_unity_client/Assets/Resources/data/.
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

    additional_assets: list[AssetSource] = []

    AssetPipeline(
        datasets=datasets,
        additional_assets=additional_assets,
        output_subdir="",
    ).process()
