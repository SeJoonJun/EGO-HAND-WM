# Assembly101 e4 oracle-geometry anticipation diagnostic

## Status and claim

This experiment is a controlled, single-egocentric-view diagnostic built on the official
Assembly101 one-second action-anticipation split. It asks whether ground-truth future camera and
hand geometry contains information that improves semantic action prediction.

The geometry-conditioned rows are **not standard action anticipation**, because they use
privileged ground-truth measurements after the official observation cutoff. Report them as:

> Oracle future-geometry diagnostics on the Assembly101-Ego e4 variant.

Only the RGB-history row obeys the official anticipation input boundary and can be compared to
ordinary anticipation methods. No predicted geometry is used in this first diagnostic.

## View selection

Each recording contributes exactly one physical headset view, e4. Depending on the capture rig,
its MP4 is named either `HMC_84358933_mono10bit` or `HMC_21179183_mono10bit`; these names are
alternative camera serials, not two simultaneous e4 inputs.

The official paper identifies e3 and e4 as the lower headset cameras and reports them as the two
strongest individual ego cameras for recognition. The official ego benchmark normally pools all
four ego cameras, however, so e4 is a defensible single-view choice rather than the community's
standard view. Never confuse ego `e4` with fixed/exocentric `v4` (`C10119_rgb`).

## Temporal protocol

Let `s` be the target action's start frame in the official 30 fps semantic annotation and
`a = s - 30` the official observation cutoff.

The semantic prediction is made at `a`: the label is the action that will begin one second later
at `s`. The one-second interval is therefore **after the prediction time but before the target
action starts**. It is not one second after action onset.

### Observed RGB

- source: e4 only;
- 32 frames sampled at 8 Hz;
- approximately four seconds of history;
- final RGB frame is exactly `a`;
- no RGB frame after `a` is visible.

At 8 Hz, the 32 endpoint-inclusive sample times relative to `a` are
`-3.875, -3.750, ..., -0.125, 0.000` seconds (a nominal four-second clip). Geometry history is
sampled at exactly these same timestamps. The official benchmark fixes the one-second
anticipation gap; this four-second history and 8 Hz sampling are our V-JEPA-aligned model choice,
not an additional Assembly101 requirement.

This follows the clip size and sampling rate in the official V-JEPA action-anticipation
configuration while retaining Assembly101's exact one-second cutoff. Raw Assembly101 video is
60 Hz, so semantic frame `f` maps to raw frame `2*f`; timestamp/PTS-based sampling should be used
when selecting the 8 Hz clip.

### Oracle future geometry

Geometry history uses the same 32 timestamps as RGB and ends at `a`. The oracle stream then has
two consecutive parts sampled at 8 Hz:

```text
pre-action gap:       a + 0.125, ..., a + 1.000 = s
target execution:     s + 0.125, ..., min(s + 1.000, e)
```

Here `e` is the annotated end of the target action. Geometry timestamps after `e` are padded and
masked rather than taken from the next action. The maximum oracle horizon is therefore two seconds
after the nominal prediction cutoff: one second leading to target onset plus up to one second of
target-action execution. This produces at most 16 future geometry tokens and 48 geometry tokens in
total when combined with the 32-token history.

The execution portion is necessary for the intended question: whether ground-truth future hand or
camera dynamics contains enough information to identify the semantic target action. Stopping at
`s` would test preparatory geometry only. Because this stream observes the target action after its
onset, it is an oracle semantic diagnostic and not a standard causal anticipation input. No
future-horizon sweep is required for the primary experiment.

## Geometry representation

Use the final observed e4 camera at `a` as anchor frame `A`. For released camera-to-world
extrinsics,

```text
T_A_C(t) = inverse(T_W_C(a)) @ T_W_C(t)
T_A_H(t) = inverse(T_W_C(a)) @ T_W_H(t)
```

Translations are converted from millimetres to metres and rotations use the continuous 6D
representation.

This matches the useful coordinate decomposition in the two reference projects. EgoVLA first
recovers world-space trajectories and then applies the inverse of the current/anchor camera to
every trajectory state. VITRA likewise applies the anchor frame's world-to-camera transform to
all wrist roots while keeping finger articulation in MANO/wrist-local coordinates. World space is
therefore the common intermediate coordinate system, not the tensor given directly to the
semantic model. Feeding raw recording-world coordinates would preserve arbitrary sequence origins
and would make the `handpose` condition contain global wrist motion.

At each future time, encode:

- camera: anchored translation (3) plus rotation-6D (6), for 9 values;
- each hand root: anchored wrist translation (3) plus rotation-6D (6), from `xf_transf`;
- each hand articulation: all 21 released `landmarks3D` points transformed into the instantaneous
  wrist frame using `inverse(T_W_H(t))`, for 63 values;
- validity: released `hand_confidences` and availability masks.

Each hand token therefore contains 72 values (9 root + 63 articulated landmarks). Hand slots are
kept as released slot 0 and slot 1 unless a handedness audit establishes a reliable left/right
mapping. Invalid hands are masked, not zero-valued and treated as observations.

This decomposition defines the requested modality ablations precisely. Each hand representation
is evaluated both alone and together with camera egomotion:

- `camera`: e4 egomotion only;
- `wrist`: the two anchored wrist SE(3) trajectories without camera;
- `handpose`: wrist-local 21-joint articulation without camera or wrist trajectory;
- `whole_hand`: wrist SE(3) plus wrist-local articulation without camera;
- `camera_wrist`: e4 egomotion plus the two anchored wrist SE(3) trajectories;
- `camera_handpose`: e4 egomotion plus wrist-local 21-joint articulation, with wrist translation
  and orientation removed from the hand-pose component;
- `camera_whole_hand`: e4 egomotion, wrist SE(3), and wrist-local 21-joint articulation.

### Matched last-observed-wrist experiment

The completed eight-row suite above keeps each wrist root in the last-observed camera frame. A
matched follow-up changes only the wrist-root reference:

```text
camera motion: T_C0_C(t) = inverse(T_W_C(0)) @ T_W_C(t)
hand-0 motion: T_H0_H(t) = inverse(T_W_H0(0)) @ T_W_H0(t)
hand-1 motion: T_H1_H(t) = inverse(T_W_H1(0)) @ T_W_H1(t)
```

Consequently, the camera and each hand root are independently identity at the legal observation
cutoff. Past timestamps describe how each wrist approached its `t=0` state; future timestamps
describe cumulative motion away from it. Finger landmarks remain instantaneous wrist-local poses.
This motion-only pose-9 representation has the same width as the completed wrist representation,
so model capacity and initialization remain matched.

Only the four wrist-containing conditions require rerunning: `wrist`, `whole_hand`,
`camera_wrist`, and `camera_whole_hand`. RGB, camera-only, and handpose-only outputs are unchanged.
At the 0.25 released-confidence threshold, 38,259/40,169 train segments and 12,323/13,140
validation segments have at least one valid wrist at `t=0`; hands without a valid legal anchor are
masked for the complete wrist-motion stream instead of being anchored to a future frame.

## Frozen RGB encoder

Use V-JEPA 2.1 ViT-G/16, 2B parameters, 384-pixel input:

```text
constructor: vjepa2_1_vit_gigantic_384
checkpoint:  vjepa2_1_vitG_384.pt
output width: 1664
```

Assembly e4 is monochrome, so repeat it over three channels before the official normalization.
Freeze the V-JEPA encoder throughout. In the primary eight-row diagnostic, disable V-JEPA's native
future-latent predictor so that the only privileged future signal is the explicitly controlled
geometry.

The 32-frame clip produces 16 temporal tubelets. Spatially pool each 24-by-24 patch grid to 4-by-4
before caching, yielding 256 visual tokens per sample. Cache last-layer features in float16 and train
only the downstream projection and probe.

## Prediction head

1. Project the 256 frozen V-JEPA tokens from 1664 to model width 512.
2. Following the supplied STA `HandPoseEncoder`, pack the selected geometry into one ordered vector
   per timestamp: camera 9-D, wrist 20-D, hand-pose 128-D, whole-hand 148-D,
   camera+wrist 29-D, camera+hand-pose 137-D, or camera+whole-hand 157-D. The final two or four
   values are explicit per-hand validity indicators.
3. Apply input LayerNorm and a linear projection to width 512. Add learned index-position,
   continuous signed physical-time, and a three-way phase embedding: observed history, oracle
   pre-action gap, or oracle target execution.
4. Process the complete ordered sequence with a masked two-layer temporal Transformer. Keep all
   output tokens rather than mean-pooling away temporal order.
5. Use three learned semantic queries (verb, object, action), following V-JEPA's attentive-probe
   design. The queries first cross-attend to visual memory and then receive a gated residual from
   cross-attention to geometry memory.
6. Apply separate linear verb, object, and action classifiers.

As in the supplied `ROIHandCrossAttention`, the final geometry residual projection is initialized
to zero. Our version contains no Faster R-CNN, FPN, RPN, ROIAlign, proposal, box, or TTC machinery:
verb/object/action queries replace the reference's ROI queries. Every row uses the same visual
path, query pooler, classifier widths, splits, and optimization; only the geometry input and mask
change.

## Primary ablations

Every geometry-conditioned model receives both the four-second observed history and the selected
oracle future stream. Paired hand-only and camera-plus-hand rows isolate the contribution of
camera motion.

| Run | Observed e4 RGB | Geometry stream, past + GT future |
|---|---:|---|
| `rgb` | yes | none |
| `rgb_gt_camera` | yes | camera |
| `rgb_gt_wrist` | yes | both wrists |
| `rgb_gt_handpose` | yes | both wrist-local 21-joint poses, without wrist trajectory |
| `rgb_gt_whole_hand` | yes | both wrists + both wrist-local poses |
| `rgb_gt_camera_wrist` | yes | camera + both wrists |
| `rgb_gt_camera_handpose` | yes | camera + both wrist-local 21-joint poses, without wrist trajectory |
| `rgb_gt_camera_whole_hand` | yes | camera + both wrists + both wrist-local poses |

The launch contains only these eight rows. There are no separate past-only, horizon-sweep,
shuffled-geometry or predicted-geometry runs.

## Training and evaluation

- official Assembly101 anticipation train and validation annotations;
- 17 verbs, 90 objects, and 1,064 actions;
- frozen V-JEPA backbone;
- train geometry encoder, attentive probe, and three heads only;
- equal-weight verb, object, and action focal losses;
- run the complete official validation split every two epochs and once on the final epoch;
- primary metric: class-mean Top-5 recall for verb, object, and action;
- also report the released tail and unseen-toy validation subsets;
- select hyperparameters on validation only.

The released pose archive has no pose streams for 14 of the 268 e4 train/validation procedures:
11 train procedures (1,780 segments) and three validation procedures (555 segments). These
segments remain in every ablation so the RGB and geometry rows use exactly the same 40,169-train
and 13,140-validation examples. Their geometry timestamps are all masked; consequently, every
geometry-conditioned model reduces to its RGB path on those samples. We do not create or tune on
a separate pose-available split.

The end-to-end manifest audit additionally found 247 labeled segments whose requested 48-step
geometry window extends beyond the released pose stream. For 242 of these, the legal observation
cutoff itself is beyond the pose stream, so all geometry is masked; five retain the available
prefix and mask only missing future timestamps. These samples also remain in all eight rows. The
remaining 50,727 segments have a complete geometry window. The machine-readable audit is in
`docs/assembly101/manifest_audit.json`.

## Local readiness

The V-JEPA code repository is `/home/jun.se/EGO-HAND-WM-REF/vjepa2`, and the official checkpoint is
stored at `/projects/torresani-lab/sejoon/checkpoints/vjepa2/vjepa2_1_vitG_384.pt`. The final launch
uses cached V-JEPA tokens and the oracle geometry model; the old DINOv3/TempAgg files are retained
only as legacy references and are not used by the launch configuration.
