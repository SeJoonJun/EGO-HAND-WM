from __future__ import annotations

import argparse
import json

from ego_hand_wm.anticipation.training import run_anticipation_training
from ego_hand_wm.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("override", nargs="*", help="Additional dotted KEY=VALUE overrides")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, [*args.overrides, *args.override])
    print(json.dumps(run_anticipation_training(config), sort_keys=True))


if __name__ == "__main__":
    main()

