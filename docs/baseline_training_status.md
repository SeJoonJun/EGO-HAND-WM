# Adapted baseline training status

Last audited: 2026-07-21 15:00 EDT

Common protocol: six observed frames, sixteen future 3-D hand positions at
30 Hz, trajectories expressed in the last-observed camera frame. Checkpoint
selection uses validation data only. Test sets are never used for selection.

## MMTwin

| Dataset | Training | Validation selection | Best checkpoint | Formal test |
|---|---|---|---|---|
| H2O | Complete, 30/30 epochs | Full 1,830-example validation split; deterministic diffusion loss. Best/final loss 0.128303 at epoch 30. | `/projects/torresani-lab/sejoon/runs/baselines/MMTwin/paper-clean-native1000-seed0-h2o/best.pt` | Pending ADE/FDE reverse-diffusion evaluation |
| EgoPAT3D | Complete, 30/30 epochs | Full 1,735-example validation split; deterministic diffusion loss. Best loss 0.175695 at epoch 28. | `/projects/torresani-lab/sejoon/runs/baselines/MMTwin/paper-clean-native1000-seed0-egopat3d/best.pt` | Pending seen/unseen ADE/FDE reverse-diffusion evaluation |

H2O continuation is optional, not required to complete the original run. A
continuation must preserve the epoch-30 checkpoint, retain Adam moments, and
restart a low learning-rate schedule; the original cosine schedule is already
at zero. Evaluate the existing best checkpoint before deciding whether the
continuation improved the paper metric.

## USST (ResNet-18)

| Dataset | Training | Validation selection | Best checkpoint | Formal test |
|---|---|---|---|---|
| H2O | Mechanically complete, but failed the baseline sanity gate | Full validation every epoch; selected epoch 2, ADE 0.174069 m | `/projects/torresani-lab/sejoon/runs/baselines/USST/H2O/paper-h6k16-res18-seed0-earlyselect-v2/snapshot/best.pth` | Measured ADE 0.171376 m / FDE 0.172569 m, but this result is rejected pending correction |
| EgoPAT3D | Incomplete; canceled during epoch 1 | No validation completed, therefore no valid best checkpoint | None in the corrected early-selection run | Pending |

On the identical H2O test manifest, last-observation persistence achieves
ADE/FDE 0.021224/0.037736 m and one-step constant velocity achieves
0.019016/0.042353 m. USST is therefore substantially worse than trivial
baselines, so its current checkpoint is not a defensible paper result. The
ten-epoch selection also ended at the end of USST's ten-epoch warmup; treating
early worsening during warmup as ordinary overfitting was incorrect.

The canceled EgoPAT3D run decoded all 22 frames even though the ResNet uses
only the six observed frames. Random MP4 seeks on the shared project filesystem
produced repeated two-to-three-minute loader stalls. The replacement loader
decodes six frames and leaves the sixteen masked future RGB slots as zeros.
Before relaunch: run loader/model equivalence tests, benchmark at least 16
batches, then run one complete epoch plus full validation. Do not launch the
20-epoch selection run until those gates pass.

## HandsOnVLM

| Dataset | Canceled progress | Durable checkpoint | Validation/test status |
|---|---:|---|---|
| H2O | 3,193/17,500 optimizer steps (18.25%) | Epoch 1 / step 1,750 | No validation-based best checkpoint; no formal test |
| EgoPAT3D | 3,164/17,500 optimizer steps (18.08%) | Epoch 1 / step 1,750 | No validation-based best checkpoint; no formal test |

Both jobs were numerically alive but produced a DeepSpeed allocator-cache flush
warning almost every optimizer step and took about 3.9 seconds per step. The
10-epoch design therefore requires roughly 19 hours per dataset on one H200.
Before relaunch: add full validation at each epoch, select `best` by validation
ADE, test resume from the epoch-1 checkpoint, and eliminate or justify the
repeated cache flushes.

## Launch gates

No paper run is submitted until all five checks pass:

1. Dataset counts and train/validation/test identities match the common manifests.
2. One real batch has shape H=6/K=16 and the last-observed-camera anchor is verified.
3. Forward, backward, checkpoint save, and checkpoint resume pass.
4. Full validation produces ADE/FDE and writes a validation-selected `best` checkpoint.
5. A measured throughput benchmark provides an ETA before the full job is queued.
