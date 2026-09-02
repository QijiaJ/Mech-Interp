# Attention-feature experiments

Code accompanying *Understanding Attention Heads*. The repository studies a
joint relation-message decomposition of Gemma-3-1B-IT attention heads. Its main
experiment assigns one latent unit to each active query event, fits a Q/K
relation subspace `U_a` and message-innovation subspace `V_a`, and compares
ordinary PCA/K-means initialization with a label-blind product-kernel spectral
initialization.

This is the compact exploratory implementation used in the writeup. It is not
the repository's separate, registered pair-level estimator or its model/rank
selection pipeline.

## Repository layout

| Path | Purpose |
|---|---|
| `data/natural_behavior_corpus.json` | Frozen source-disjoint corpus: 250 TRAIN and 250 HOLD contexts with L1H1 and L5H2 active events. |
| `eda/` | Natural-corpus construction and label-aware diagnostics used only to motivate or interpret the label-blind fits. |
| `code/algorithm1.py` | Earlier learned positive-Q/K, rank-one pair co-clustering baseline. |
| `code/algorithm2.py` | Event-level `R + W_joint` hard-EM fit with PCA/K-means initialization. |
| `spectral/code/spectral_cluster.py` | Product-kernel spectral discovery, eigengaps, anchor clustering, and Nyström extension. |
| `code/algorithm2_spectral_init.py` | The same event-level EM initialized by spectral labels. |
| `code/algorithm2_downstream.py` | Spectral-start fitting plus causal downstream intervention; saves fitted projectors. |
| `code/algorithm2_downstream_from_fits.py` | Audited Q/K-routed interventions from saved fits. |
| `code/investigate_label_algorithm_2.py` | K-means and known-centroid initialization diagnostic. |
| `code/plot_algorithm2_initialization_geometry.py` | Data for the three-dimensional initialization-geometry plot. |
| `tests/` | Focused corpus, geometry, and spectral-kernel tests. |

Generated JSON, plots, checkpoints, and reports belong under `outputs/` or
`artifacts/`; both are ignored by Git.

## Requirements

- Python 3.11 or newer.
- The dependencies pinned in `requirements.txt`.
- A checkout of the parent Mech-Interp project, specifically its
  `unifying_algorithm/unifying_attention` package. That package supplies the
  authenticated Gemma extraction and model-loading code.
- The pinned `google/gemma-3-1b-it` snapshot used by the corpus.
- A fresh, successfully validated synthetic smoke report from the same
  Mech-Interp source checkout. Real-head scripts fail closed without it.

Set the project locations and install dependencies:

```bash
export ATTENTION_FEATURE_ROOT=/absolute/path/to/attention_feature_code
export MECH_INTERP_ROOT=/absolute/path/to/Mech-Interp

python -m pip install -r "$ATTENTION_FEATURE_ROOT/requirements.txt"
python -m pip install -e "$MECH_INTERP_ROOT/unifying_algorithm"
```

Create and validate the current-source smoke from `unifying_algorithm/` before
loading Gemma:

```bash
cd "$MECH_INTERP_ROOT/unifying_algorithm"
python -m unittest discover -s tests -v
python -m unifying_attention.experiments.unlearned_projector_smoke \
  --output-dir /tmp/unlearned_projector_em_smoke
```

The smoke report used below is:

```text
/tmp/unlearned_projector_em_smoke/unlearned_projector_em_smoke.json
```

## Verify this repository

```bash
cd "$ATTENTION_FEATURE_ROOT"
python -m unittest discover -s tests -v
```

The retained suite contains 16 tests. The tests do not load the model.

## Reproduce the writeup pipeline

All commands below use the frozen corpus and write only beneath `outputs/` or
`artifacts/`.

### 1. Label-aware EDA

This step produces the TRAIN/HOLD Q/K and message-subspace diagnostics used to
state the descriptive `L1H1: A=2` and `L5H2: A=1` hypotheses. Labels are not
passed to any subsequent fit.

```bash
cd "$ATTENTION_FEATURE_ROOT"
python eda/candidate_head_ground_truth_eda.py \
  --smoke-report /tmp/unlearned_projector_em_smoke/unlearned_projector_em_smoke.json \
  --corpus data/natural_behavior_corpus.json \
  --output-dir outputs/eda \
  --output outputs/eda.json
```

### 2. PCA/K-means initialization and event-level EM

This is the K-means-start comparison in the writeup. Defaults are relation rank
32 and message rank 64.

```bash
python code/algorithm2.py \
  --repo-root "$MECH_INTERP_ROOT" \
  --smoke-report /tmp/unlearned_projector_em_smoke/unlearned_projector_em_smoke.json \
  --corpus data/natural_behavior_corpus.json \
  --output outputs/algorithm2_kmeans.json
```

The initialization-only diagnostic is:

```bash
python code/investigate_label_algorithm_2.py \
  --repo-root "$MECH_INTERP_ROOT" \
  --smoke-report /tmp/unlearned_projector_em_smoke/unlearned_projector_em_smoke.json \
  --corpus data/natural_behavior_corpus.json \
  --output outputs/kmeans_diagnostic.json
```

### 3. Product-kernel spectral discovery

This independently computes the anchor affinity, normalized spectral embedding,
Nyström HOLD assignment, and eigengaps for `A=1,...,4`. The implementation uses
at most six deterministic anchor events per source and the eight highest-
attention non-BOS source positions per event.

```bash
python spectral/code/spectral_cluster.py \
  --repo-root "$MECH_INTERP_ROOT" \
  --smoke-report /tmp/unlearned_projector_em_smoke/unlearned_projector_em_smoke.json \
  --corpus data/natural_behavior_corpus.json \
  --output-dir outputs/spectral
```

### 4. Spectral-start event-level EM

This changes only the TRAIN initializer: the subsequent `R + W_joint` M- and
E-steps are the same as in Algorithm 2.

```bash
python code/algorithm2_spectral_init.py \
  --repo-root "$MECH_INTERP_ROOT" \
  --smoke-report /tmp/unlearned_projector_em_smoke/unlearned_projector_em_smoke.json \
  --corpus data/natural_behavior_corpus.json \
  --output outputs/algorithm2_spectral.json
```

The writeup's fixed hypotheses are `A=2` for L1H1 and `A=1` for L5H2, with
`r=32`, `c=64`, spectral seed 1729, 20 deterministic K-means restarts, and no
refitting on HOLD.

### 5. Context-specific downstream interpretation

The first command refits the fixed spectral-start models, runs the interventions,
and stores the projectors. The second command reloads those projectors and
recomputes the audited Q/K-only HOLD interventions used for the representative
examples.

```bash
python code/algorithm2_downstream.py \
  --repo-root "$MECH_INTERP_ROOT" \
  --smoke-report /tmp/unlearned_projector_em_smoke/unlearned_projector_em_smoke.json \
  --corpus data/natural_behavior_corpus.json \
  --artifact-dir artifacts/algorithm2_fits \
  --output outputs/algorithm2_downstream.json

python code/algorithm2_downstream_from_fits.py \
  --repo-root "$MECH_INTERP_ROOT" \
  --smoke-report /tmp/unlearned_projector_em_smoke/unlearned_projector_em_smoke.json \
  --corpus data/natural_behavior_corpus.json \
  --artifact-dir artifacts/algorithm2_fits \
  --output outputs/algorithm2_downstream_audited.json
```

## Earlier learned-feature baseline

The writeup retains the learned positive-Q/K rank-one pair method only as an
earlier baseline. Run it separately; it is not interchangeable with the
event-level estimator above.

```bash
python code/algorithm1.py \
  --repo-root "$MECH_INTERP_ROOT" \
  --smoke-report /tmp/unlearned_projector_em_smoke/unlearned_projector_em_smoke.json \
  --corpus data/natural_behavior_corpus.json \
  --output outputs/algorithm1_pair_baseline.json
```

## Regenerating the frozen corpus

The checked-in corpus is sufficient for the experiments above. To rebuild it,
obtain the public HeadVis Gemma sequence export and run:

```bash
python eda/freeze_natural_behavior_corpus.py \
  --smoke-report /tmp/unlearned_projector_em_smoke/unlearned_projector_em_smoke.json \
  --headvis-sequences /absolute/path/to/headvis_gemma3_sequences.json \
  --output data/natural_behavior_corpus.json
```

This step selects sources and events deterministically, recomputes activations
on the pinned IT checkpoint, and enforces source-disjoint TRAIN/HOLD splits.

## Evidence boundary

`TRAIN` and `HOLD` here are the development splits used by the accompanying
writeup. The scripts explicitly mark locked and historical test access as false.
The fitted ranks are fixed rather than selected, and the spectral eigengap is an
exploratory nomination of `A`, not a complete model-selection procedure.
