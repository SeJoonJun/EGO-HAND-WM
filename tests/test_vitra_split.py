import json
import io
import tarfile
from pathlib import Path

import numpy as np
import torch
import yaml

from ego_hand_wm.contracts.batch import canonical_collate
from ego_hand_wm.data.synthetic import SyntheticCanonicalDataset
from ego_hand_wm.data.build import VitraShardDataset
from ego_hand_wm.data.vitra_split import VitraVideoSplit, episode_member_identity
from ego_hand_wm.models.world_action_model import WorldActionModel
from ego_hand_wm.training.validation import evaluate_vitra


def test_episode_member_identity_for_all_sources() -> None:
    examples = {
        "ego4d_cooking_and_cleaning/episodic_annotations/Ego4D_abc_ep_000001.npy": (
            "ego4d_cooking_and_cleaning",
            "abc",
        ),
        "ego4d_other/episodic_annotations/Ego4D_def_ep_000002.npy": (
            "ego4d_other",
            "def",
        ),
        "egoexo4d/episodic_annotations/EgoExo4D_take_1_ep_000003.npy": (
            "egoexo4d",
            "take_1",
        ),
        "epic/episodic_annotations/epic_kitchens_P01_01_ep_000004.npy": (
            "epic",
            "P01_01",
        ),
        "ssv2/episodic_annotations/somethingsomethingv2_123_ep_000000.npy": (
            "ssv2",
            "123",
        ),
    }
    for member, expected in examples.items():
        assert episode_member_identity(member) == expected


def test_split_holds_out_shared_ego4d_physical_video(tmp_path: Path) -> None:
    cooking_member = (
        "ego4d_cooking_and_cleaning/episodic_annotations/Ego4D_shared_ep_000001.npy"
    )
    other_member = "ego4d_other/episodic_annotations/Ego4D_shared_ep_000002.npy"
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": 42,
                "dataset_aliases": {"ego4d_other": "ego4d_cooking_and_cleaning"},
                "validation_videos": {"ego4d_cooking_and_cleaning": ["shared"]},
                "validation_members": {
                    "ego4d_cooking_and_cleaning": [cooking_member],
                    "ego4d_other": [other_member],
                },
            }
        )
    )
    split = VitraVideoSplit.load(path)
    assert not split.includes(
        "train",
        logical_source="ego4d_other",
        video_name="shared",
        member_name=other_member,
    )
    assert split.includes(
        "validation",
        logical_source="ego4d_other",
        video_name="shared",
        member_name=other_member,
    )
    assert not split.includes(
        "validation",
        logical_source="ego4d_other",
        video_name="shared",
        member_name="ego4d_other/episodic_annotations/Ego4D_shared_ep_999999.npy",
    )
    assert split.episode_seed(other_member) == split.episode_seed(other_member)


def test_validation_reports_each_physical_source_and_macro() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/smoke.yaml").read_text())
    config["model"].update({"hidden_dim": 32, "depth": 1})
    dataset = SyntheticCanonicalDataset(
        length=4,
        history_steps=3,
        future_steps=4,
        horizon_seconds=1.0,
        image_size=16,
        seed=17,
    )
    logical_sources = (
        "ego4d_cooking_and_cleaning",
        "egoexo4d",
        "epic",
        "ssv2",
    )
    samples = []
    for index, source in enumerate(logical_sources):
        sample = dataset[index]
        sample["metadata"] = {
            "archive_member": f"{source}/episode-{index}.npy",
            "source_dataset": source,
            "horizon_seconds": 1.0,
        }
        samples.append(sample)
    batch = canonical_collate(samples)
    metrics = evaluate_vitra(
        WorldActionModel(config["model"]),
        [batch],
        device=torch.device("cpu"),
        use_bf16=False,
        ode_steps=1,
        ode_method="euler",
        visual_normalization="none",
        visual_normalization_eps=1e-6,
    )
    for source in ("ego4d", "egoexo4d", "epic", "ssv2"):
        assert f"validation/{source}/camera_translation_cm" in metrics
    assert "validation/macro/camera_translation_cm" in metrics
    assert "validation/micro/camera_translation_cm" in metrics


def test_validation_members_are_filtered_before_numpy_decode(tmp_path: Path) -> None:
    selected = "epic/episodic_annotations/epic_kitchens_P01_01_ep_000001.npy"
    rejected = "epic/episodic_annotations/epic_kitchens_P02_01_ep_000001.npy"
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": 42,
                "dataset_aliases": {},
                "validation_videos": {"epic": ["P01_01"]},
                "validation_members": {"epic": [selected]},
            }
        )
    )
    payload = io.BytesIO()
    np.save(payload, {"video_name": "P01_01"}, allow_pickle=True)
    shard = tmp_path / "episodes.tar"
    with tarfile.open(shard, "w:") as archive:
        for name, data in ((rejected, b"not a numpy payload"), (selected, payload.getvalue())):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    dataset = VitraShardDataset(
        {
            "shard_glob": str(shard),
            "sampling": {"kind": "native_variable", "history_steps": 2},
            "split_manifest": str(split_path),
            "split": "validation",
        }
    )
    episodes = list(dataset._episodes(dataset.shards))
    assert len(episodes) == 1
    assert episodes[0][0] == selected
    assert episodes[0][1]["video_name"] == "P01_01"
