# VITRA dual-expert training from scratch

## Decision

Train the improved dual-expert model from random initialization. The run retains the future
visual expert, adds explicit wrist-local 21-joint/fingertip supervision from the first step, and
does not load the completed 200k checkpoint. No long run should be submitted until the matching
full-model batch benchmark succeeds.

## Batch semantics

The completed run used:

```text
16 samples/GPU x 4 GPUs x accumulation 4 = global batch 256
```

The preferred candidate is:

```text
64 samples/GPU x 4 GPUs x accumulation 1 = global batch 256
```

This increases the physical microbatch but does not increase the effective global batch. It
therefore preserves the optimizer and learning-rate semantics. It also preserves total FLOPs per
optimizer step: the speedup comes from better batched kernels and eliminating three extra
forward/backward launches, not from doing one quarter of the mathematical work.

The old run used about 26.7 GiB per 143.8-GiB H200, so batch 64 is plausible, not guaranteed.
The full dual-expert benchmark compares 16x4, 32x2, and 64x1 at the same global batch and must
measure peak memory and wall time before the production run.

## Scratch architecture and objective

The scratch configuration uses the complete architecture:

```text
observed DINO + text + camera/wrist/MANO history
                         |
                 shared context encoder
                         |
                shared per-layer K/V cache
                    /                 \
        geometry flow expert      visual flow expert
          camera/wrist/MANO        future DINO tokens
                  |
        21-joint hand decoder
```

The visual and geometry future experts do not cross-attend one another. Both train the shared
history representation through the shared context cache. Geometry uses uniform flow-time
sampling. Visual flow alone uses the DINO-space time shift from Flow World Models (shift 13).

The geometry objective combines flow matching, factorized rigid/MANO geodesic losses,
camera-hand decomposition, wrist-local 21-joint error, fingertip error, and timestamp-aware joint
velocity. The auxiliary hand decoder is zero initialized to repeat the last observed wrist-local
pose, so it starts from a meaningful baseline even though all model weights are trained from
scratch.

## Schedule

The old generated-validation curve still improved materially through roughly 150k steps and then
flattened around 160k-180k. Therefore, 30k is not a defensible scratch-training cap. The scratch
config uses 180k as a safety maximum, validates every 2k steps, keeps the best model by generated
fingertip MPJPE, and stops after ten validations without a 0.01-cm improvement. The endpoint is
selected by validation rather than assuming all 180k steps are necessary.

## Launch gate

1. Submit `slurm/benchmark_vitra_scratch_batches.sbatch` only when four H200s are available.
2. Select the fastest finite-loss arm with at least 10% peak-memory headroom.
3. Run a short four-GPU smoke that includes validation and checkpoint reload.
4. Only then submit `slurm/train_vitra_visual_kinematics_scratch.sbatch`.

Neither script is submitted automatically.
