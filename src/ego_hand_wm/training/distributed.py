from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as distributed


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize(device_setting: str = "auto") -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_setting == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    elif device_setting in ("auto", "cpu"):
        device = torch.device("cpu")
    else:
        raise RuntimeError(f"Requested {device_setting} but CUDA is unavailable")
    if world_size > 1 and not distributed.is_initialized():
        distributed.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return DistributedContext(rank, world_size, local_rank, device)


def seed_everything(seed: int, rank: int = 0) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup() -> None:
    if distributed.is_available() and distributed.is_initialized():
        distributed.destroy_process_group()

