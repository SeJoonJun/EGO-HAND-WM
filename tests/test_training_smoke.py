from pathlib import Path

import yaml

from ego_hand_wm.training.engine import run_training


def test_one_step_training(tmp_path: Path) -> None:
    root = Path("/n/home08/sjmathy/EGO-HAND-WM")
    config = yaml.safe_load((root / "configs/smoke.yaml").read_text())
    config["model"]["hidden_dim"] = 32
    config["model"]["heads"] = 4
    config["model"]["depth"] = 1
    config["data"]["length"] = 4
    config["data"]["image_size"] = 16
    config["training"]["output_dir"] = str(tmp_path / "run")
    config["training"]["max_steps"] = 1
    result = run_training(config)
    assert result["step"] == 1
    assert (tmp_path / "run/last.pt").is_file()

