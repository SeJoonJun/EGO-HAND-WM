from __future__ import annotations

import argparse
import json

from ego_hand_wm.config import load_config
from ego_hand_wm.training.engine import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("override", nargs="*", help="Additional dotted KEY=VALUE overrides")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, [*args.overrides, *args.override])
    result = run_training(config)
    if int(result["step"]) > 0:
        print(json.dumps({"status": "complete", **result}))


if __name__ == "__main__":
    main()

