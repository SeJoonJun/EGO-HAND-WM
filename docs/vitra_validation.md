# VITRA validation protocol

The upstream VITRA human-pretraining release does not provide an official train/validation
partition. Its released dataset wrapper passes no `training_path`, so all entries in each
`episode_frame_index.npz` are used by training. We therefore label this repository's split as a
project-specific protocol rather than an official VITRA benchmark split.

`configs/splits/vitra-small-validation-v1.json` is deterministic (seed 42) and leakage-safe:

- The unit of holdout is the physical video UUID/take, never an episode.
- The same held-out Ego4D UUID set is excluded from both `ego4d_cooking_and_cleaning` and
  `ego4d_other`, because those archives share one physical video pool.
- Training excludes every episode from every held-out physical video.
- Recurring validation uses a bounded deterministic subset: 32 episodes from each Ego4D archive
  and 64 each from EgoExo4D, EPIC, and SSv2 (256 total).
- The sampled history/future window for a validation episode is seeded from its archive member
  name, so it is stable across epochs and runs.

The actual physical-group counts are recorded in the generated manifest. SSv2 can require fewer
held-out videos than evaluated episodes because some physical clips contribute multiple episodes;
selection stops after reaching the per-source evaluation target.

Report metrics for the four physical sources (`Ego4D`, `EgoExo4D`, `EPIC`, and `SSv2`) and the
unweighted macro mean over those four sources. A pooled micro average may be included as a
secondary diagnostic, but must not replace the macro mean because source sizes differ sharply.
