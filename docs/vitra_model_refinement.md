# VITRA world-action model refinement

## Decision

Continue from the completed 200k visual+geometry checkpoint with a short geometry-only
refinement stage. The refinement adds explicit wrist-local 21-joint and fingertip supervision,
keeps the strong rigid-pose branch, and starts a fresh optimizer/schedule. Do not resume the
zero-ended 200k scheduler.

The production config is `configs/vitra_geometry_kinematics_refine.yaml`; its launcher is
`slurm/train_vitra_kinematics_refine.sbatch`. Batch size 64 is a high-confidence H200 candidate,
but must pass the supplied 16/32/64 throughput array before a long run.

## Evidence from the completed run

The 200k model beats repeat-last persistence for rigid geometry but not articulation:

| Metric | Model | Persistence | Relative result |
|---|---:|---:|---:|
| Camera translation | 1.530 cm | 2.051 cm | 25.4% better |
| Camera rotation | 2.010 deg | 2.586 deg | 22.3% better |
| Wrist translation | 4.078 cm | 6.535 cm | 37.6% better |
| Wrist rotation | 13.537 deg | 16.944 deg | 20.1% better |
| MANO rotation | 10.221 deg | 9.701 deg | 5.4% worse |

The fixed 256-episode validation split further shows why the average MANO objective is weak:

| Repeat-last wrist-local diagnostic | Value |
|---|---:|
| 21-joint MPJPE | 0.651 cm |
| Fingertip MPJPE | 1.396 cm |
| 90th-percentile joint displacement | 1.806 cm |

Fingertips move more than twice as far as the average joint, but the old loss averages all 15
MANO rotations within each hand and then averages those two hand terms with camera and wrists.
After multiplying by `rotation_weight=0.1`, each hand's direct geodesic contribution is only
0.02 of its unweighted value.

## Refined architecture

The pretrained path stays unchanged:

```text
6 observed frames
  ├─ 17 DINO.txt tokens/frame
  ├─ 13 geometry entities/frame
  ├─ text + intrinsics
  └─ physical-time embeddings
          ↓
6-layer bidirectional context encoder
          ↓
per-layer observed-context K/V cache
          ↓
12-layer future geometry flow expert
  (future queries attend [observed K/V + future-geometry K/V])
          ├─ canonical camera / wrists / MANO velocity heads
          └─ new 3D hand-kinematics head
```

For each future step and hand, the new head takes that hand's wrist token plus its five
finger-chain tokens. Twenty-one learned joint queries cross-attend those six anatomical tokens,
pass through a small joint transformer, and predict a residual over the last observed
wrist-local 21-joint pose. Its output projection is zero initialized, so the exact initial
solution is repeat-last persistence rather than an arbitrary skeleton.

The targets are computed directly from released VITRA geometry:

```text
p_local(t) = R_wrist_world(t)^T [p_world(t) - t_wrist_world(t)]
```

This removes camera and global wrist motion from the articulation target. It matches VITRA's own
`joints_manospace` construction and its supported 21x3 keypoint action representation.

The refined objective is:

```text
L = L_flow
  + 0.1 L_rigid_rotation
  + 0.5 L_MANO_rotation
  + 0.2 L_camera_hand_decomposition
  + 10.0 L_21_joint
  + 10.0 L_fingertip
  + 0.05 L_joint_velocity
```

Joint losses use robust Smooth-L1. Velocity uses the actual future timestamps and includes the
last history pose as t=0. This makes the objective valid for both consecutive and sparsely sampled
future windows.

## Why these references were used

- [DexWM](https://github.com/facebookresearch/dexwm) adds an explicit hand-consistency decoder
  because future visual-feature regression alone does not preserve manipulation geometry. Its
  decoder is 2D heatmap based; this project already owns metric 3D geometry, so the appropriate
  analogue is wrist-local 3D full-joint/fingertip supervision.
- [VITRA](https://github.com/microsoft/VITRA) natively supports wrist/MANO-space 21x3 joints as an
  action representation. The new target uses the same coordinate construction instead of
  introducing an incompatible convention.
- [Flow World Models](https://github.com/facebookresearch/Flow-World-Models) uses shifted flow-time
  sampling in high-dimensional DINO feature space. The code now supports an independent visual
  flow-time distribution, so a future joint pretraining run can shift visual time without changing
  geometry time.
- [FastWAM](https://github.com/yuantianyuan01/FastWAM) samples video and action noise times with
  separate schedulers. This supports decoupling the two modalities rather than forcing the same
  random time as the old implementation did.

## Training-speed changes

The completed resume ran 75k optimizer steps in about 14 h 37 min (approximately 0.702 s/step),
with effective global batch 256 and only about 26.7 GiB used per H200. Mean GPU utilization was
about 84%, so checkpoint/validation stalls were not the main cost. The important changes are:

1. **Geometry-only refinement.** The 245.6M-parameter future visual expert is omitted after it has
   shaped the shared context. This should be the largest compute reduction.
2. **One microbatch per optimizer step.** `64/GPU x 4 GPUs x accumulation 1` preserves global batch
   256, replacing four sequential 16-sample microbatches. The provided array measures 16x4,
   32x2, and 64x1 before committing.
3. **Fused AdamW and DDP bucket views.** Enabled for the H200 run; static graph and buffer broadcast
   controls remove avoidable DDP overhead.
4. **FP16 host feature transfer.** Frozen DINO context is already stored in FP16. The refinement
   keeps it FP16 through host transfer and projects it under BF16 autocast on GPU.
5. **Persistent workers and early split filtering.** Held-out members are rejected from tar paths
   before NumPy decoding; workers stay alive across epochs/validations.
6. **Fresh short schedule with early stopping.** At most 30k steps, LR 5e-5 with a 0.1 floor, best
   checkpoint selected by macro MANO rotation, patience eight validations.
7. **Fewer full checkpoints.** Full resume state every 10k; compact model-only weights every 2.5k
   and whenever validation improves. The old run wrote roughly 277 GiB because each full Adam
   checkpoint was about 6.8 GiB.

## Required gates before the long run

1. Run `slurm/benchmark_vitra_refine_batches.sbatch` when four H200s are genuinely available.
2. Choose the fastest setting that has finite loss and at least 10% memory headroom at K=24.
3. Run a 100-step four-GPU smoke with validation and confirm all ranks log identical initialization
   missing/unexpected-key counts.
4. Compare 5k refinement validation against both the 200k checkpoint and repeat-last. Stop if
   rigid metrics regress materially without a MANO/fingertip gain.
5. Report generated four-step Heun metrics, not teacher-forced auxiliary losses, for model claims.

## Ablations needed for a paper claim

Use the same initialization, global batch, data order, and validation split:

1. old objective, geometry-only refinement;
2. factorized MANO rotation only;
3. factorized rotation + 21-joint loss;
4. factorized rotation + 21-joint + fingertip loss;
5. full objective including timestamp-aware joint velocity.

This separates gains from dropping the visual expert, stronger MANO weighting, full-hand geometry,
and the fingertip-specific signal.
