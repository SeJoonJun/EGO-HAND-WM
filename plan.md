# EGO-HAND-WM — Positioning & Plan

Status: architecture revised 2026-07-18 for the licensed DINO.txt representation; temporal/data
plan revised 2026-07-17 after paper + code audits of
USST, MMTwin, Uni-Hand, EggHand, EgoH4, EgoMAN, EgoVLA, VITRA, Being-H0.7, Donk, DexWM,
EgoWAM, FlowWM, plus direct audits of the VITRA and EgoVLA human/robot data pipelines and a
stratified timing/language audit of 6,400 released VITRA episodes using their cached video PTS.
Every comparison cell below is backed by a paper read or a local code audit. Marks: ✓ yes ·
✗ no · △ partial (footnote).

## Headline claim (qualifier-exact form)

> The first **single** model, with **one explicit geometric output space**
> (camera SE(3) + wrist SE(3) + MANO articulation), that is **competitively evaluated on
> human-centric egocentric forecasting benchmarks** *and* **transfers to dexterous robot
> manipulation**.

Each qualifier eliminates a family: *single* → Uni-Hand's per-dataset/per-task checkpoints;
*explicit geometric output incl. camera* → everyone (see egomotion column below); *human-centric
benchmarks evaluated* → EgoVLA/VITRA/Being-H0/EgoWAM/DexWM; *dexterous* → Uni-Hand (gripper) and
EgoWAM (parallel jaw).

## Table A — Predicted output space (what each model outputs and is evaluated on)

| Model | Wrist pos | Wrist rot | MANO/articulation | Camera SE(3) out | Contact | Future visual | Egomotion handling |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| USST (ICCV'23, 2307.08243) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | oracle odometry as input |
| MMTwin (IROS'25, 2504.07375) | ✓ | ✗ | ✗ | △¹ | ✗ | ✗ | latent homography, never decoded |
| Uni-Hand (T-PAMI'26, 2511.12878) | ✓ | ✗ | ✗ | △¹ | ✓ | ✗ | latent; dead code in release |
| EggHand (2605.07642) | ✓² | ✗ | ✓ (42 joints) | ✗ | ✗ | ✗ | canonicalized away |
| EgoH4 (2504.08654) | ✓² | ✗ | ✓ (joints) | ✗ | ✗ | ✗ | future camera poses given as input |
| EgoMAN (2512.16907) | ✓ | ✓ | ✗ | ✗ | △³ | ✗ | not modeled |
| EgoVLA (2507.12440) | ✓ | ✓ | ✓ (MANO-15) | ✗ | ✗ | ✗ | consumed in preprocessing, discarded |
| VITRA (2510.21571) | ✓ | ✓ | ✓ (MANO-45) | ✗ | ✗ | ✗ | absorbed into camera-frame labels |
| Being-H0.7 (2605.00078) | ✓ | ✓ | ✓ | ✗ | ✗ | △⁴ | not modeled |
| Donk (2606.03868) | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ video | not modeled |
| DexWM (2512.13644) | ✗⁵ | ✗ | ✗ | ✗ | ✗ | ✓ latents | conditioning input, zeroed at planning |
| EgoWAM (2607.08436) | ✓ | ✓ | ✗ (gripper) | ✗ | ✗ | △⁴ | factored out via oracle Aria VIO |
| **Ours** | **✓** | **✓** | **✓** | **✓** | ✗⁸ | **△⁴ optional** | **predicted as first-class output** |

## Table B — Model & evaluation settings

| Model | Single generalizable model | Language | RGB-only inference | Human-centric benchmarks | Robot manipulation |
|---|:-:|:-:|:-:|:-:|:-:|
| USST | ✗ per-dataset | ✗ | ✗ (3D traj input) | ✓ | ✗ |
| MMTwin | ✗ per-dataset | ✗ | ✗ (voxels, GLIP) | ✓ | ✗ |
| Uni-Hand | ✗ per-dataset/task | △⁶ | ✗ (RGB-D + voxels) | ✓ | △ gripper |
| EggHand | ✓ | ✓ | △ (camera poses in) | ✓ | ✗ |
| EgoH4 | ✓ | ✗ | ✗ (future poses in) | ✓ | ✗ |
| EgoMAN | ✓ | ✓ | ✓ | ✓ | ✗ |
| EgoVLA | ✓ | ✓ | △ (proprio) | ✗ | ✓ dexterous |
| VITRA | ✓ | ✓ | △ (hand state) | ✗ | ✓ dexterous |
| Being-H0.7 | ✓ | ✓ | △ | ✗ | ✓ dexterous |
| Donk | ✓ | ✓ | ✓ | △ own protocols | ✗ |
| DexWM | ✓ | ✗ | ✗ (needs future actions) | ✗ | ✓ planning⁷ |
| EgoWAM | ✓ | ✗ | △ (proprio) | ✗ | △ gripper |
| **Ours** | **✓** | **✓** | **✓ (input dropout)** | **✓** | **✓ dexterous** |

## Footnotes

1. Egomotion exists only as an undecoded latent conditioner — never output, never evaluated.
   Functional in MMTwin (loss weight 5e-4, latent space); constructed-but-never-called in the
   Uni-Hand release (motion features hard-coded to zeros).
2. Wrist position via predicted joint positions; no explicit wrist rotation output.
3. Approach/manipulation stage labels, not contact states.
4. Training-time auxiliary only, discarded at inference (EgoWAM world head; Being-H0.7 latent
   alignment; our parallel future-DINO expert grounded in the shared observed-context cache).
5. DexWM consumes hand keypoints + camera 6-DoF as its *action conditioning*; its only pose-like
   output is 2D keypoint heatmaps decoded from predicted latents.
6. Text conditioning claimed in the paper; absent from the released code.
7. CEM/MPC planning with a dexterous Allegro hand — no learned policy; requires goal images and
   ground-truth future actions to roll the world model forward.
8. Deliberately dropped (2026-07-15): contact adds annotation pipelines and is Uni-Hand's
   contribution; grasp timing on the robot side comes from MANO articulation (EgoVLA needs no
   contact head). Cost: no Uni-Hand CE-metric comparison. Revisit only if grasp-timing analysis
   becomes a reviewer issue.

## Caption argument (how to read the tables)

Table A: the "Camera SE(3) out" column is empty except for ours, and the egomotion-handling
column enumerates **seven distinct avoidance strategies** (oracle input, latent-undecoded,
canonicalized, given-as-input, discarded, conditioning-only, VIO-factored) — the field treats
egomotion as important enough to handle, but never as a prediction target. Table B: every
competitor has ≥1 ✗ in the {single model, human-centric benchmarks, dexterous manipulation}
conjunction — the forecasting line (top block) fails generality and robots; the VLA/WAM line
(bottom block) fails human-centric evaluation. Ours is the only row that closes the conjunction.

## Contribution stack (ranked, verified open)

1. **Formulation**: explicit future camera SE(3) as a jointly predicted, explicitly evaluated
   output alongside wrist SE(3) + MANO — untaken across papers *and* code.
2. **Mechanism**: hard SE(3) camera/hand decomposition plus a differentiable camera-relative
   supervision loss: `T_Ct<-Wt = inverse(T_A<-Ct) @ T_A<-Wt`, where `A = C0` is the final
   observed camera. This couples predicted egomotion to predicted hand motion and is
   inexpressible in a single-stream wrist predictor.
3. **Capability**: one checkpoint across both regimes (human-centric forecasting + dexterous
   manipulation transfer) + released protocol/canonicalization infrastructure.
4. **Analysis**: egomotion-decomposition evaluation (world-frame vs camera-frame hand error via
   the model's own predicted ego) — no baseline can replicate.

Demoted to cited machinery (do NOT claim): continuous time queries (NeRMo, ALIEN), visual-future
training prior (InternVLA-A1.5, Being-H0.7, EgoWAM, FlowWM), retargeting pipeline (EgoVLA),
contact head (Uni-Hand), masked multi-dataset training.

## Final architecture (v1, frozen 2026-07-18)

The v1 model is a **geometry-first conditional world-action model**, not an EgoVLA clone with a
camera head and not a pixel video generator. The primary backbone uses DINOv3 for visual
features, the aligned DINO.txt text tower, a multimodal context transformer, and a joint
conditional flow-matching trajectory denoiser. A PaliGemma/VLM context encoder is retained only
as an ablation/fallback; it is not the default architecture.

### 1. Canonical future state

Let the final observed camera be the anchor frame `A = C0`. At every future physical query time
`t_k`, predict:

```text
x(t_k) = {
    T_A<-C(t_k),                 # camera trajectory
    T_A<-W_left(t_k),            # left wrist trajectory
    T_A<-W_right(t_k),           # right wrist trajectory
    R_MANO_left(t_k, 15 joints), # full local articulation
    R_MANO_right(t_k, 15 joints)
}
```

Network-facing representations:

- camera: translation-3 + rot6D-6;
- each wrist: translation-3 + rot6D-6;
- each hand: 15 local joint rotations × rot6D-6;
- left/right validity masks at every history and query time;
- optional MANO shape `beta` as metadata/conditioning, never as a predicted dynamic state.

Full MANO local rotations, rather than PCA-15, are the canonical articulation. VITRA's released
left pose is preserved in its self-consistent MANO_RIGHT-derived local basis and distinguished by
left-specific entity embeddings; it must not be mislabeled as native MANO_LEFT. Conversion to
native MANO_LEFT or robot kinematics belongs in the deterministic output adapter. For EgoVLA
metric compatibility, project/refit the full pose to its PCA-15 basis.

Camera-relative wrist pose is derived, not independently invented:

```text
T_Ct<-Wt = inverse(T_A<-Ct) @ T_A<-Wt
```

This derived quantity receives camera-relative supervision whenever annotations exist. It is the
central differentiable coupling between egomotion and hand motion.

### 2. Conditioning inputs

Use a variable-length observation window. For the initial VITRA gate, use up to six genuine
native annotation frames with exact PTS; do not claim that six frames represents one second
(it spans about 0.08 s for 60 Hz EPIC, 0.17 s for 30 Hz Ego4D/EgoExo4D, and 0.42 s for 12 Hz
SSv2). Downstream observation windows are specified in physical seconds:

- RGB observation history;
- language instruction;
- past left/right wrist SE(3);
- past full MANO articulation;
- past camera trajectory relative to `C0`, when available;
- current/past left/right validity masks;
- normalized camera calibration token `(fx/W, fy/H, cx/W, cy/H)`;
- physical timestamp in seconds for every observation and future query.

All conditioning streams support dropout. This enables matched-input baseline comparisons,
missing-modality training, and an RGB+text-only inference ablation. Future camera, wrist, MANO,
or RGB information is never used as input.

Do **not** feed raw robot qpos or dataset identifiers into the shared backbone. Dataset and
embodiment differences live at deterministic boundary adapters. The only embodiment-specific
information permitted is inside the robot input/output adapter, outside the shared model.

### 3. Backbone blocks

```text
RGB history -> frozen DINOv3 + DINO.txt head --+
                                                |
Text -> cached frozen DINO.txt text tower -------+--> context fusion transformer
                                                |
Past canonical geometry + timestamps -----------+
Camera calibration token -----------------------+
                                                        |
                                                        v
                                  observed world-context transformer
                                                        |
                                              shared per-layer K/V cache
                                                /                 \
                                               v                   v
                              geometry flow expert       future-DINO flow expert
                             (train + inference)             (training only)
```

Initial implementation:

- DINOv3 ViT-L/16 plus the official two-layer DINO.txt vision head, both frozen;
- per frame, retain the post-head class token and 4x4 equal-area pooled post-head patch tokens
  (`17 x 1024` total); reconstruct the exact aligned global descriptor as the normalized
  concatenation of class token and mean spatial token (`2048` dimensions);
- frozen DINO.txt text tower, cached once as one normalized 2048-D conditioning token per exact
  VITRA prompt (T5 remains an ablation, not the production default);
- trainable multimodal world-context transformer;
- cacheable per-layer context K/V shared by both future experts;
- DiT-style conditional geometry flow expert;
- 13 structured future entity tokens per query: camera, left/right wrist, and five contiguous
  three-joint MANO kinematic-chain tokens per hand;
- stream masks/gates so unavailable outputs do not contribute to attention or loss.

Every future expert layer performs one masked softmax over its own K/V concatenated with the
cached observed-context K/V. Context queries never attend future tokens. Geometry queries attend
only observed context and geometry; visual queries attend only observed context and visual
tokens. Geometry and future visual tokens therefore never exchange information.

Only if language grounding is a measured bottleneck should a VLM replace the context encoder.
That experiment must leave the canonical state and flow trajectory model unchanged.

### 4. Physical time

Use only the selected LaWAM mechanism: every state token receives a sinusoidal embedding of its
physical timestamp `physical_time = t_k` in **seconds**. A query set is defined jointly by its
actual timestamps and its valid token count; token index alone never implies either horizon or
frequency.

The code must keep two unrelated time variables separate:

- `physical_time`: where an observation/target lies in the sequence;
- `flow_time`: the flow-matching noise/denoising coordinate, injected through AdaLN.

Never reuse one embedding for the other. Variable query counts are padded/masked per batch. Use
native video PTS and annotation timestamps rather than assuming that every VITRA video is exactly
30 Hz.

The frozen temporal capability contract is:

| Profile | Valid future queries | Physical timing | Purpose |
|---|---:|---|---|
| VITRA parity gate | 16 | consecutive native annotation PTS | reproduce/debug released behavior |
| VITRA production pretraining | core `K in {4,8,12,16}`; rare `K=24` | genuine native PTS, dense or temporally spread | query-count/density diversity without fabricated labels |
| EgoVLA robot post-training | 30 | exactly `1/30, ..., 30/30` s | released 1 s, 30 Hz action chunk |
| Human-centric post-training | up to 60 | exactly `1/30, ..., 60/30` s when GT exists | up to 2 s at 30 Hz |

`K=32` is excluded from regular VITRA sampling: the audit found primary-language support of only
2--12% outside EPIC (EPIC: 71%), so using it routinely would turn token-count diversity into an
EPIC/source-length bias. `K=24` is capped at approximately 10% and source-balanced. At least 90%
of VITRA production samples use `K<=16`.

For VITRA production, 70% of query sets use consecutive native targets (dense local dynamics) and
30% use sorted, temporally spread genuine targets from the same valid action interval (horizon
diversity). Always retain the furthest selected target, attach exact relative PTS, and never
interpolate, duplicate an episode boundary, or silently cross a hand-language interval. A longer
benchmark horizon requires downstream examples at that horizon; timestamp embeddings alone do
not justify extrapolation to 5 s.

### 5. Training-only future visual latent expert

Pixels are not generated in v1. Use a parallel future-DINO expert:

```text
observed world-context K/V -> geometry flow expert
                           -> future-DINO flow expert
```

Both future experts read the same observed-context K/V but cannot attend each other. The visual
loss therefore trains the context encoder and cache projections directly, while future visual
targets can never leak into geometry. At inference the visual expert and its noisy tokens are not
instantiated; the imagination-trained context encoder remains and its K/V is prefetched once for
all geometry-flow steps.

Use frozen post-DINO.txt-head targets with one class token plus a 4x4 spatial grid (`17 x 1024`)
for every future frame. The visual expert is a separate time-conditioned transformer with the
same depth as the geometry expert. It uses independent Gaussian noise and the shared context
cache. Keep the geometry-only recipe as an ablation, and retain the visual objective in the final
recipe only if it improves held-out geometry metrics. Geometry-only inference never instantiates
or executes future visual queries.

## Objective

The primary objective is per-stream-normalized, validity-masked conditional flow matching:

```text
L_FM = sum_s (1 / D_s) * || M_s * (v_pred_s - v_target_s) ||^2
```

where `s` is camera, wrist, or MANO; `D_s` is that stream's dimension; and `M_s` contains data,
hand-reconstruction validity, per-hand language-interval validity, query/padding validity, and
temporal masks. Normalize each stream by its own valid-token count before combining streams;
otherwise dense EPIC samples and longer query sets dominate shorter sources. Dimension
normalization prevents the full-MANO stream from overwhelming the camera/wrist streams.

Decode the estimated clean endpoint and add small physically meaningful auxiliary losses:

```text
L_total = L_FM
        + lambda_rot    * L_SO3_geodesic
        + lambda_fk     * L_MANO_joints_and_fingertips
        + lambda_ego    * L_camera_hand_decomposition
        + lambda_visual * L_future_DINO_FM
```

- `L_SO3_geodesic`: camera, wrist, and MANO local-joint rotations after rot6D projection;
- `L_MANO_joints_and_fingertips`: MANO FK joint/tip error in metric coordinates;
- `L_camera_hand_decomposition`: error of the derived `T_Ct<-Wt` against camera-relative GT;
- `L_future_DINO_FM`: optional training-only latent flow loss.

Suggested starting coefficients after per-stream whitening are `lambda_rot=0.1`,
`lambda_fk=0.1`, `lambda_ego=0.2`, and `lambda_visual=0` followed by a ramp to at most `0.1`.
They are initialization values, not claims; select them on validation geometry, never visual loss.

## Data compatibility and adapters

### VITRA-1M adapter

VITRA contains enough information for the canonical state: World2Cam extrinsics, intrinsics,
wrist world translation/orientation, full 15-joint local MANO rotations, MANO shape, 21 world and
camera joints, frame indices, text, and per-hand validity.

For every sampled window:

1. Recover physical time from video PTS/frame mapping.
2. Set `A = C0` at the final observed frame.
3. Transform all history/future wrist and joint geometry from world into `A`.
4. Compute `T_A<-Ct = E0 @ inverse(Et)` from the per-frame camera extrinsics.
5. Convert wrist/camera/MANO rotations to canonical rot6D.
6. Preserve VITRA's released right-canonical local articulation for both hands and retain side
   identity through distinct left/right entity embeddings. Do not apply an unverified reflection.
7. Resolve the active left and right text intervals independently at the anchor; text ranges are
   half-open episode-frame intervals `[start, end)` and can have different boundaries per hand.
8. Build independent per-query masks for left wrist/MANO and right wrist/MANO by combining that
   hand's `kept_frames`, action-language interval, and episode-boundary validity. Camera validity
   is independent of both hands.
9. Preserve original/GPT-rephrased text variants without leaking a future action label into an
   action-anticipation input.
10. Extract globally deduplicated frozen DINO.txt visual targets once (`17 x 1024`, float16) and
    read them directly by physical frame ID for both context and the auxiliary future branch.
11. Enumerate every anchor-dependent VITRA prompt and cache its frozen normalized DINO.txt text
    embedding once.

The final tensor length is shared inside a sample/batch, but left and right valid counts may
differ. Never require both hands to be annotated. In the 6,400-episode audit, every episode had
primary-hand text while only 9.5--31.7% had secondary-hand text; among episodes with both texts,
their temporal intervals overlapped approximately 90--96%, but their reconstruction masks still
differed.

Measured primary-hand support after reserving six history frames was:

| Source | `K=4` | `K=8` | `K=12` | `K=16` | `K=24` | `K=32` |
|---|---:|---:|---:|---:|---:|---:|
| Ego4D cooking/cleaning | 100% | 100% | 79.7% | 56.1% | 21.6% | 6.1% |
| Ego4D other | 100% | 100% | 81.7% | 58.8% | 25.6% | 10.4% |
| EPIC | 100% | 100% | 100% | 100% | 100% | 71.2% |
| EgoExo4D | 100% | 100% | 86.8% | 65.2% | 30.1% | 12.2% |
| SSv2 | 100% | 100% | 68.9% | 39.8% | 10.9% | 2.3% |

These percentages are sampler constraints, not dataset-performance claims. The sampler must
resample a smaller supported `K` rather than pad a requested unsupported target with a duplicated
boundary state.

VITRA's instantaneous `joints_camspace(t)` is expressed in `Ct`; it is not automatically an
entire trajectory expressed in `C0`. The annotations contain everything needed for this
canonicalization, but the window-dependent `C0` transformation still must be performed.

### EgoVLA robot adapter

The audited EgoVLA release makes human and robot samples compatible only after preprocessing. Its
released shared input per hand is camera-frame wrist translation-3, wrist rotvec-3, and five
fingertip xyz positions-15. Raw robot qpos never enters the shared prediction head directly.

Our robot boundary is:

```text
robot qpos + EEF + fingertips
    -> EgoVLA camera conversion and robot-to-MANO fitting
    -> canonical wrist SE(3) + side-specific full MANO + fingertips
    -> shared backbone
```

The sequential HDF5 data permit past proprioception histories even though the released EgoVLA
decoder consumes only current proprioception. Expand EgoVLA PCA coefficients through its
side-specific MANO layer to create full local rotations. Follow the offline/training handedness
convention, not the apparent `is_right=True` left-hand typo in the online evaluation helper.

The EgoVLA simulation policy uses a fixed camera. Supply identity past camera motion and mask or
freeze the future-camera loss/head during robot post-training; do not train the shared camera
predictor toward static identity on a large robot mixture.

Robot execution reuses the released boundary controllers:

```text
predicted wrist SE(3) -> EgoVLA differential IK
predicted full MANO -> MANO FK five fingertips -> EgoVLA hand_actuation_net
```

No new IK/controller is required. EgoVLA robot fine-tuning makes only the first six PCA
coefficients effective even though it reserves PCA-15 per hand; our full-MANO model should mask
uncontrollable robot articulation directions rather than pretending they are independently
supervised.

### Other human datasets

- HOT3D: supervise wrist, MANO/fitted hand geometry, and dynamic camera motion.
- GigaHands: supervise wrist and MANO; enable camera loss only after per-frame egomotion and
  calibration are verified. Otherwise mask it rather than inserting zero motion.
- EgoDex: camera and wrist supervision where its calibration/trajectory passes the adapter gates.
- EgoPAT3D-DT/H2O-PT/EgoMAN protocols: use benchmark-specific I/O adapters without changing the
  canonical backbone representation.

## Training pipeline

### Stage 0 — representation gates (mandatory before large training)

Implement and test:

1. VITRA canonicalize/decanonicalize round trip.
2. Left/right chirality and MANO model/version checks.
3. MANO rotations -> FK joints/fingertips reconstruction.
4. EgoVLA robot qpos/EEF -> MANO/fingertips round trip.
5. Wrist prediction -> EgoVLA IK/control dry run.
6. Camera transform identity/composition tests.
7. Independent left/right `kept_frames` plus language-interval boundary tests, including windows
   where one hand changes action before the other.
8. Units, rotation convention, fingertip ordering, masks, and PTS tests.
9. A fixed-16 native-time VITRA parity test before enabling variable query sampling.

Do not start the full VITRA run until these gates pass on visual overlays and numeric tolerances.

### Stage 1 — VITRA-1M pretraining

- Train the context fusion transformer and geometry flow denoiser.
- Keep DINOv3 and the text encoder frozen initially.
- Condition on RGB, text, and available past geometry.
- Supervise future camera, wrists, and full MANO with stream masks.
- Apply random input-modality dropout.
- First pass the released-behavior parity gate with 16 consecutive native future annotations.
- For production, sample `K in {4,8,12,16}` from genuine future annotations; permit `K=24` in at
  most approximately 10% of source-balanced samples and do not use `K=32` regularly.
- Use 70% consecutive-native and 30% sorted temporally-spread query sets. Attach exact PTS and
  apply independent left/right reconstruction + language-interval masks.
- Normalize every stream loss by its valid-token count and use query dropout so the denoiser
  cannot infer timing from tensor length or position.
- First establish geometry-only performance; then test the future-DINO auxiliary.

### Stage 2 — human-centric joint post-training

- Jointly post-train on HOT3D, GigaHands, and other validated human datasets.
- Retain one canonical checkpoint and dataset-specific loaders only.
- Use balanced sampling and masked supervision rather than fake labels.
- Train the standard human forecasting request up to 60 queries over 2 s at 30 Hz only on clips
  with matching genuine supervision; shorter/sparser datasets supervise their available query
  times without fabricated intermediate targets.
- Fine-tune evaluation baselines on the same train splits and timing protocol for the matched-data
  comparison; report their native/pretrained setting separately.

### Stage 3 — EgoVLA robot post-training

- Initialize from the VITRA/human checkpoint; zero-shot robot use is not claimed.
- Convert every robot sequence through the canonical adapter.
- Fine-tune the exact EgoVLA temporal interface: 30 future actions at 30 Hz over 1 s.
- Train wrist and controllable articulation outputs; mask/freeze camera prediction.
- Use a smaller shared-backbone learning rate.
- Mix approximately 10–20% human replay batches or use representation distillation so robot
  training does not erase human forecasting/camera knowledge.
- Fine-tune the robot boundary adapter and upper denoiser blocks first; unfreeze deeper blocks
  only if validation success requires it.
- Deploy with closed-loop/receding-horizon replanning: predict 30 actions, execute one environment
  step, observe again, and replan through EgoVLA IK and hand actuation.

The final deliverable is one post-trained checkpoint serving both human forecasting and robot
manipulation, with output streams enabled/masked according to the task.

## Evaluation plan

### Human-centric forecasting

- wrist translation ADE/FDE;
- wrist rotation geodesic error;
- MANO joint, fingertip, and vertex error;
- camera ATE/RPE and rotation error;
- camera-compensated versus raw egocentric hand error;
- multimodal flow samples under the exact benchmark convention.

Primary data: HOT3D joint-state protocol and GigaHands hand protocol. External validation:
EgoMAN-Bench and the EgoPAT3D-DT/H2O-PT protocols runnable through USST/MMTwin. Report 0.5 s,
1 s, and 2 s wall-clock-aligned horizons where GT exists, with 2 s as the maximum standard
single-pass human request. Add longer horizons only with matching training examples and an
explicit rolling-generation protocol.

### Robot manipulation

Use EgoVLA's Ego Humanoid Manipulation Benchmark and its released retargeting/controller stack.
All compared methods receive the same observations, action horizon, replanning frequency, IK,
retargeting, and success evaluator. The standard action request is exactly 30 predictions over
1 s at 30 Hz, replanned every environment step. Report aggregate and per-task success.

### Baselines and fairness

- Translation forecasting: USST, MMTwin, HandsOnVLM, constant velocity.
- Full wrist SE(3)/MANO: EgoVLA, VITRA, Being-H0.7 where runnable.
- Manipulation: EgoVLA/VITRA-style released runnable systems; never invent numbers for unreleased
  Donk variants.
- Regime A: matched train/eval data and inputs; every baseline is adapted/fine-tuned to the same
  protocol.
- Regime B: native recipes/checkpoints versus our single unified checkpoint, clearly labeled.
- Validate baseline adapters by reproducing published metrics before using cross-method results.
- Check and disclose any EgoVLA/HOT3D train-split overlap.

### Required ablations

1. No camera prediction.
2. Independent camera head without geometric coupling.
3. No past geometry/proprioception.
4. PCA-15 versus full MANO rotations.
5. Deterministic regression versus flow matching.
6. Future-DINO auxiliary off/on.
7. Separate human/robot checkpoints versus the unified post-trained checkpoint.
8. EgoVLA plus a naive camera head.
9. DINOv3+text context versus a VLM context encoder.
10. Fixed 16 consecutive VITRA targets versus the production variable-`K` native-PTS sampler.

## Falsification and build order

- Phase 0: finish downloads, canonical adapters, tests, and baseline inference harness.
- Phase 1: reproduce USST/MMTwin published anchors on EgoPAT3D-DT/H2O-PT.
- Phase 2: measure EgoVLA zero-shot and matched-finetuned performance on human forecasting.
- Phase 3: run an explicit-egomotion ablation against no-ego and latent-ego variants.
- Phase 4: overfit a small VITRA subset, then train the geometry-only VITRA model.
- Phase 5: test the future-DINO auxiliary only after Phase 4 is stable.
- Phase 6: robot GT round-trip/controller test, followed by robot post-training.
- Pilot baseline: EgoVLA plus a camera head, which directly tests and preempts the claim that the
  full proposal is merely EgoVLA with one extra output.

The core proposal is falsified or must be narrowed if explicit egomotion fails to improve hand
forecasting/decomposition, if the unified checkpoint cannot retain human performance after robot
post-training, or if the future-DINO auxiliary does not improve geometry metrics.

## Final model claim

> One geometry-first, physically time-conditioned world-action model jointly predicts egomotion,
> wrist SE(3), and full hand articulation; it is pretrained on human egocentric video, evaluated
> on human forecasting, and transferred through deterministic embodiment adapters to dexterous
> robot manipulation.

## Timing

Donk (NeurIPS'26 sub) is competition only on the MANO axis — cite + differentiate (no ego, no
robot, no code, 17-frame horizon). Speed matters: a Donk v2 could add camera on a Wan backbone.
Decision point late August with Phase 0–3 numbers in hand: tightly-scoped ICLR'27 (forecasting +
ego ablation + pilot transfer) vs CVPR'27 (full manipulation suite). arXiv on submission day
either way to timestamp the ego claim.
