# NanowireML — ESM-2-Primary Methodology Review Document

**Status:** Draft for peer review (AI coding agents + self-review) prior to preprint submission
**Target venue:** IEEE BIBM workshop paper or bioRxiv preprint
**Repo:** https://github.com/moneet-dev/nanowire-ml
**Last updated:** 2026-07-01

> **Purpose of this document.** This is a standalone methodology + rationale record,
> written to be handed to an AI coding agent (or a human reviewer) with no other
> context, so it can (a) verify the code matches the stated methodology, (b) flag
> statistical or biological reasoning errors, and (c) sanity-check that reported
> numbers are trustworthy before this becomes a submitted paper. Every claim below
> is either already implemented in the linked script, or explicitly marked TODO.
> Result sections are intentionally left blank — they are filled in after the
> Colab run completes and numbers are copied over verbatim (no rounding beyond
> what the scripts already do).

---

## 1. Background

### 1.1 The biological problem

Certain bacteria (*Geobacter*, *Shewanella*, sulfate-reducing bacteria) perform
**extracellular electron transfer (EET)**: they move electrons from internal
metabolism to external solid acceptors (iron oxides, electrodes) via conductive
protein filaments called **microbial nanowires**. Two structurally unrelated
protein families both build nanowires:

- **e-Pili** (Type IV pili, e.g. PilA): short (60–90 aa), aromatic-residue
  stacking carries charge.
- **Multiheme cytochromes** (e.g. OmcS, OmcZ, MtrC): long (>200 aa), heme
  cofactors bound via CXXCH motifs carry charge.

These two classes share **no detectable sequence or structural homology** —
they converged on the same function independently. This is why **BLAST and
other alignment-based annotation tools fail** to identify novel nanowire
proteins: there is nothing to align against. A sequence-only ML classifier
that generalizes across both structural classes is the only tractable
computational screening approach, and this has real economic stakes: SRB-driven
microbially induced corrosion (MIC) costs the US oil & gas industry an
estimated $1B/year, and engineered nanowires are being explored for microbial
fuel cells, biosensors, and biodegradable bioelectronics.

Full biological background: [`P1_NanowireML_background.md`](../../P1_NanowireML_background.md).

### 1.2 The paper we reproduce and extend

**Raya D, Peta V, Bomgni A, Do TD, Kalimuthu J, Salem DR, Gadhamshetty V,
Gnimpieba EZ, Dhiman SS.** *Prediction and validation of nanowire proteins in
Oleidesulfovibrio alaskensis G20 using machine learning and feature
engineering.* Comput Struct Biotechnol J. 2025;27:1706–1718.
[doi:10.1016/j.csbj.2025.04.022](https://doi.org/10.1016/j.csbj.2025.04.022)

This is the peer-reviewed version of a 2023 bioRxiv preprint
(doi:10.1101/2023.05.03.539336) from **Dr. Gnimpieba's lab at USD** — the lab
this work targets as a PhD pre-application artifact. The paper:

- Built a binary classifier: nanowire protein vs. RecA (negative control,
  unrelated cytoplasmic DNA-repair protein)
- Used 21 iFeature descriptor types → 19,857 hand-crafted features
- Reported: SVM/RF/XGBoost/LR/MLP with accuracy 0.9487–0.9669, ROC-AUC
  0.9696–0.9920 (Table 2)
- **Did not report:** PR-AUC, calibration (ECE), uncertainty quantification,
  a complete feature ablation, or any protein-language-model baseline

### 1.3 Why this project exists (the gap)

Six gaps identified against the original paper (see `P1_architecture.md` §1
and `P1_NanowireML_background.md` §5) motivate this project's contributions:

| Gap | What's missing in Raya et al. 2025 | Our addition |
|---|---|---|
| G1 | Thin baselines, no systematic tuning | RandomizedSearchCV, stacking ensemble |
| G2 | No calibration assessment | ECE, reliability diagrams, Platt/isotonic |
| G3 | No uncertainty quantification | Bootstrap 95% CIs on every headline metric |
| G4 | Incomplete feature ablation | Full per-descriptor + combination ablation |
| G5 | Negative set (RecA) may be too easy | Flagged as a limitation (§7) |
| G6 | No protein-language-model baseline | **ESM-2 embeddings — this document's focus** |

### 1.4 The pivot to ESM-2-primary (this document's scope)

Initial implementation extracted all 22 iFeature descriptors as the primary
feature set and treated ESM-2 as a secondary "stretch goal" comparison
(consistent with `P1_architecture.md` Phase 5, which frames ESM-2 as
"an alternative to iFeature"). During execution, three observations motivated
promoting ESM-2 to the **primary** feature set, with iFeature demoted to
**baseline comparison**:

1. **iFeature extraction is fragile and version-dependent.** iFeatureOmegaCLI
   on Colab's Python 3.14 environment only exposes 19/22 descriptors (KNN,
   ZScale, AAIndex unavailable — see `src/feature_utils.py::_resolve_descriptors`),
   requires a monkeypatch for an O(8000²) performance bug in TPC normalization
   (`_fast_tpc`), and has a silent-failure mode where a missing descriptor
   leaves stale data from the previous descriptor rather than raising an error
   (`_MIN_FEATURES` validation guard exists specifically to catch this).
2. **ESM-2 produces better-calibrated predictions with a fraction of the
   engineering surface.** A verified run (2026-07-01, NR variant, 5-fold CV)
   confirmed ESM-2 + Logistic Regression achieves ECE = 0.0002
   (`results/metrics/esm2_cv_table.csv`), against iFeature's best ECE = 0.0042
   for LR on the same variant (`results/metrics/reproduction_table_nr.csv`) —
   roughly a 20x improvement in calibration, with 1,280 learned features vs.
   18,510 hand-crafted ones. Both feature sets separate the classes almost
   perfectly (ROC-AUC ≈ 1.0000 for both), so calibration — not discrimination
   — is where the two approaches are actually distinguishable.
3. **Literature supports this as a real trend, not an artifact of this
   dataset.** Recent protein-language-model benchmarking work reports the same
   pattern — ESM-2 embeddings paired with a simple classifier (often LR)
   outperform hand-crafted descriptor sets on calibration and/or accuracy for
   binary protein-function classification tasks structurally similar to ours:
   - Mall et al. 2024, *Sci Rep* — ESM2-based LightGBM classifiers beat
     hand-crafted-feature baselines by 3–7 points across AUPR/AUC/F1 for
     protein crystallization prediction.
   - Chu et al. 2024, *IJMS* — fine-tuned ESM-2 embeddings outperformed PSSM
     features for druggable-protein classification (95.11% accuracy).
   - Du et al. 2023, *Food Chemistry* (pLM4ACE) — of 65 classifier×embedding
     combinations tested for antihypertensive peptide screening, **logistic
     regression with ESM-2 embeddings was the single best performer.**

   These are analogous binary protein-classification problems (not nanowire
   proteins specifically), so they support the *general trend* — PLM
   embeddings + simple linear classifier calibrate and generalize well — not
   a nanowire-specific claim. This is stated as supporting evidence, not proof.

**Framing for the paper:** ESM-2 is the primary result; iFeature reproduction
is retained in full (it is still required to validate against Raya et al.
Table 2) and serves as the traditional-ML comparison arm. The core empirical
claim becomes: *does a general-purpose protein language model, with no
nanowire-specific feature engineering, match or exceed a 19,857-feature
hand-crafted pipeline on this task — and if so, by how much, and is the
difference statistically defensible?*

---

## 2. What the code does

### 2.1 Pipeline overview

```
Phase 1 (01_data_prep.py)
  └─ merged.fasta + labels.csv  (NR: 1664 seqs, redundant: 1997 seqs)
         │
         ├──────────────────────┬──────────────────────────────┐
         ▼                      ▼                               │
Phase 2 (04b_esm2_pipeline.py)  Phase 3 (02_feature_extraction.py +
  ESM-2 embeddings (PRIMARY)     03_reproduce_baseline.py +
  - 5-fold CV classifier battery  04_extended_pipeline.py)
  - all-33-layer probing            iFeature descriptors (BASELINE)
    + bootstrap 95% CI              - Raya et al. Table 2 reproduction
  - UMAP visualization              - tuning + calibration + SHAP
  - OOF predictions saved           │
         │                          │
         └──────────┬───────────────┘
                     ▼
     Phase 4 (04c_esm2_analysis.py)
     Aligned 5-fold CV comparison
     McNemar's test + bootstrap AUC-diff CI
     ESM-2 + SHAP-top-50-iFeature combined experiment
                     │
                     ▼
     Phase 5 (05_ablation.py) — iFeature descriptor-group ablation
                     │
                     ▼
     Phase 6 (06_autoresearch_loop.py) — keep/revert experiment loop
```

### 2.2 Scripts and their exact responsibilities

| Script | Role | Key outputs |
|---|---|---|
| [`scripts/01_data_prep.py`](../scripts/01_data_prep.py) | FASTA merge, dedup, sanity checks | `data/raw/merged.fasta`, `labels.csv`, `data_sanity.csv` |
| [`scripts/04b_esm2_pipeline.py`](../scripts/04b_esm2_pipeline.py) | **Primary**: ESM-2 embed + CV battery + layer probe + UMAP | `esm2_cv_table.csv`, `esm2_layer_table.csv`, `esm2_cv_preds.csv` |
| [`scripts/02_feature_extraction.py`](../scripts/02_feature_extraction.py) | iFeature 22-descriptor extraction | `master_features.csv` (18,510 features, 19/22 descriptors) |
| [`scripts/03_reproduce_baseline.py`](../scripts/03_reproduce_baseline.py) | Reproduces Raya et al. Table 2 | `reproduction_table.csv` |
| [`scripts/04_extended_pipeline.py`](../scripts/04_extended_pipeline.py) | Tuning + calibration + stacking + SHAP on iFeature | `extended_table.csv`, `shap_top100.csv` |
| [`scripts/04c_esm2_analysis.py`](../scripts/04c_esm2_analysis.py) | Aligned statistical comparison | `comparison_table.csv`, `mcnemar_result.csv`, `auc_diff_ci.csv` |
| [`scripts/05_ablation.py`](../scripts/05_ablation.py) | iFeature descriptor-group ablation | `ablation_table.csv` |
| [`scripts/06_autoresearch_loop.py`](../scripts/06_autoresearch_loop.py) | Predefined keep/revert experiment loop on `src/classify.py` | `autoresearch_log.csv` |

### 2.3 What Phase 2 (ESM-2, primary) specifically computes

From `scripts/04b_esm2_pipeline.py`:

1. **Embedding generation** — `esm2_t33_650M_UR50D` (650M params, 33 layers,
   1280-dim output), mean-pooled over residue positions (excluding BOS/EOS
   tokens), sequences truncated to 1022 residues (ESM-2's max), batch size 4.
   All 33 layers' hidden states are extracted in a **single forward pass**
   (`repr_layers=list(range(1,34))`) — zero additional GPU cost for probing
   every layer vs. probing only the last one.
2. **Classifier battery** (5-fold stratified CV, `random_state=42`) — LR, RF,
   XGBoost, XGBoost+Isotonic calibration, and a 2-model Stacking ensemble
   (RF+XGBoost → LR meta-learner), all on the layer-33 embedding.
3. **Layer-wise probing** — an independent LR classifier fit on each of the
   33 layers' embeddings, 5-fold CV each, with **1000-resample bootstrap 95%
   CIs** on both ROC-AUC and PR-AUC per layer (added 2026-07-01; see §4.3).
4. **UMAP** — 2D projection of layer-33 embeddings, coloured by class label,
   to visually confirm separability independent of any classifier.
5. **OOF (out-of-fold) predictions saved** for the best-ECE model, to feed
   the McNemar test in Phase 4 without re-training.

### 2.4 What Phase 4 (statistical comparison) specifically computes

From `scripts/04c_esm2_analysis.py`:

1. Re-runs LR and XGBoost on **both** ESM-2 and iFeature features using the
   **same `StratifiedKFold(5, random_state=42)` object** — this is the key
   methodological fix: earlier versions used a 70/30 split for ESM-2 and
   5-fold CV for iFeature/ablation, which are not statistically comparable.
2. **McNemar's test** (continuity-corrected chi-squared, or exact binomial
   when disagreement count < 10) comparing ESM-2+LR's OOF predictions against
   iFeature+LR's OOF predictions — tests whether the two classifiers'
   *error patterns* differ significantly, not just their aggregate accuracy.
3. **Bootstrap 95% CI on the ROC-AUC difference** (ESM-2+LR − iFeature+LR)
   via 2000 paired resamples, with a one-sided p-value (P(iFeature ≥ ESM-2)).
4. **Combined experiment**: ESM-2's 1280 features concatenated with the
   top-50 SHAP-ranked iFeature features (from Phase 3's `shap_top100.csv`),
   evaluated with the same aligned CV — tests whether hand-crafted features
   add anything on top of the learned representation.

---

## 3. Methodology

### 3.1 Dataset

Two variants, both built by `scripts/01_data_prep.py` from the official
`bicbioeng/nanowire-protein-prediction` repo (mirrors Mendeley
doi:10.17632/vdvdrj3k2p.1):

| Variant | Positive | Negative | Total | Purpose |
|---|---|---|---|---|
| Non-redundant (NR) | 847 | 817 | 1,664 | Primary analysis set — exact-duplicate sequences removed |
| Redundant | 998 | 999 | 1,997 | Paper-faithful set — matches Raya et al. 2025 exactly, for direct Table 2 comparison |

One orphan FASTA header (NCBI record missing its `>` prefix) is auto-repaired
in `src/data_utils.py::repair_fasta_text`. 8 sequences contain non-canonical
residues (X/B/Z/U/O/\*), handled by skipping affected positions rather than
dropping the sequence.

**Rationale for using NR as primary:** 333 exact-duplicate sequences exist in
the redundant set (1997 − 1664). Training and evaluating on data containing
duplicates inflates apparent performance if any duplicate pair straddles a
CV fold boundary. This is a real risk here specifically: the 2023 bioRxiv
preprint of this same paper reported 99.99% XGBoost accuracy on the redundant
set vs. 96.65% in the 2025 published version — a swing large enough to be
explained by redundancy effects. We report both variants for transparency but
treat NR as the primary result.

**TODO / open question:** we deduplicate on *exact* sequence match. We do not
run CD-HIT at a similarity threshold (e.g. 90% identity), which `P1_architecture.md`
Phase 1 originally specified. Near-duplicate (non-exact) sequences may still
inflate CV performance. This is disclosed as a limitation (§7) rather than
fixed, given the scope of this pass — flag if reviewers consider it blocking.

### 3.2 Feature extraction

**ESM-2 (primary):** `esm2_t33_650M_UR50D`, no fine-tuning — used purely as a
fixed feature extractor. Embeddings are the mean over per-residue hidden
states at each layer (excluding special tokens), producing a
1280-dimensional vector per sequence per layer.

**iFeature (baseline):** 22 descriptor groups via `iFeatureOmegaCLI`
(AAC, GAAC, CKSAAP, DPC×2, TPC×2, CTD×3, CTriad, KNN, Geary, Moran, NMBroto,
AC, CC, SOCNumber, QSOrder, PAAC, ZScale, AAIndex). On the Colab runtime used
for this study, 19/22 descriptors are available (KNN, ZScale, AAIndex are
absent from that iFeatureOmegaCLI build — see `src/feature_utils.py`'s
`_resolve_descriptors` runtime probe), yielding 18,510 concatenated features
after dropping constant/duplicate columns.

### 3.3 Evaluation protocol

- **Cross-validation:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
  used identically across ESM-2, iFeature, and ablation experiments (as of the
  2026-07-01 alignment fix in `04c_esm2_analysis.py` — see §1.4 and §4.2).
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and Expected
  Calibration Error (ECE, 10 equal-width bins) — computed by
  `src/eval_utils.py::full_eval`.
- **Uncertainty:** percentile bootstrap (1000–2000 resamples,
  `random_state=42`) on ROC-AUC and PR-AUC for every reported model, via
  `src/eval_utils.py::bootstrap_metric_ci`.
- **Significance testing:** McNemar's test between the two best classifiers'
  out-of-fold predictions (ESM-2+LR vs. iFeature+LR), plus a bootstrap CI on
  the paired ROC-AUC difference (both in `04c_esm2_analysis.py`).

### 3.4 Layer-wise probing protocol (this session's addition)

- All 33 ESM-2 transformer layers' representations are extracted in one
  forward pass per batch (no repeated inference).
- An independent Logistic Regression probe (`C=1.0`, `solver=saga`,
  `max_iter=2000`, `StandardScaler` prefix) is fit on each layer's embedding
  separately, under the same 5-fold CV as the main analysis.
- Bootstrap 95% CIs (1000 resamples) are computed per layer for both ROC-AUC
  and PR-AUC, plotted as shaded bands around the per-layer point estimate.
- **Why LR specifically for probing (not XGBoost/RF):** probing experiments
  in the representation-learning literature standardly use a *linear* probe
  precisely because linear separability is the property being measured — a
  non-linear probe (tree ensemble) can partially compensate for a
  poorly-organized embedding space, which would obscure the "which layer
  encodes the signal most directly" question the experiment is designed to
  answer.

### 3.5 Statistical comparison protocol (ESM-2 vs. iFeature)

- **Same-fold constraint:** both feature sets are evaluated inside the
  identical `StratifiedKFold` object/seed, so paired statistical tests (which
  assume the two prediction sets come from the same partition of the same
  samples) are valid.
- **McNemar's test** is chosen over a paired t-test on fold-level AUCs because
  it operates directly on binary correct/incorrect predictions per sample,
  which is the more standard test for "are two classifiers' error patterns
  different" in the ML literature, and does not require the CV fold scores to
  be approximately normal (only 5 folds — too few for a reliable normality
  assumption).
- **Bootstrap AUC-difference CI** is reported alongside McNemar's test because
  McNemar's test answers a *discrete-prediction* question (0.5 threshold) while
  ROC-AUC is a threshold-free ranking metric — the two tests answer related
  but distinct questions, and reporting both avoids over-claiming from either
  alone.

---

## 4. Rationale behind each choice

### 4.1 Why ESM-2 (`esm2_t33_650M_UR50D`) and not a larger/smaller ESM-2 variant

650M is the largest ESM-2 checkpoint that comfortably fits a T4's 16GB VRAM
at batch_size=4 with room for the classifier battery running concurrently in
the same session. Smaller checkpoints (`t12_35M`, `t30_150M`) exist but the
literature comparisons cited in §1.4 predominantly use 650M-class models,
making this the more literature-comparable choice. **TODO:** an ablation
across ESM-2 checkpoint sizes (35M/150M/650M) would strengthen the "why 650M"
claim but is out of scope for this pass — flag as a possible reviewer request.

### 4.2 Why 5-fold CV instead of the original 70/30 split for ESM-2

The first ESM-2 implementation used `train_test_split(test_size=0.30,
random_state=6)` to match the iFeature baseline script's split convention.
This was changed because: (a) a single split gives one point estimate with no
internal variance signal, (b) it made ESM-2 results *structurally
incomparable* to the ablation study (which always used 5-fold CV), and (c) it
prevented a valid McNemar's test against iFeature, since McNemar's test needs
predictions for every sample, not just a 30% held-out slice.

### 4.3 Why bootstrap CIs specifically (not analytic/DeLong CIs for AUC)

DeLong's method gives an analytic CI for a single ROC-AUC and is
asymptotically exact, but (a) extending it to PR-AUC is non-trivial (no
closed form as clean as DeLong's for ROC), and (b) we already have a
bootstrap CI implementation (`eval_utils.py::bootstrap_metric_ci`) used
consistently across the whole codebase (extended pipeline, ablation, ESM-2
battery) — reusing one method everywhere is preferable to mixing an analytic
method for AUC with a resampling method for PR-AUC, for consistency of
interpretation across all reported intervals in the paper.

### 4.4 Why McNemar's test rather than a 5-fold paired t-test

A paired t-test over 5 fold-level AUC differences is a common alternative,
but with only 5 folds the test has very low power and a shaky normality
assumption. McNemar's test instead pools all N out-of-fold predictions into
one 2×2 contingency table (classifier A right/B wrong vs. A wrong/B right),
giving much higher effective sample size for the significance test. This is
standard practice for "does classifier A beat classifier B on this dataset"
questions in the classifier-comparison literature (Dietterich 1998).

### 4.5 Why iFeature is retained (not dropped) despite the ESM-2 pivot

Three reasons: (1) reproducing Raya et al. 2025's Table 2 is a required,
independently valuable deliverable regardless of which method is "better" —
it validates that the reimplementation pipeline (data loading, splitting,
metric computation) is correct before any new claim is trusted; (2) the
ESM-2-vs-iFeature comparison *is itself* the paper's central empirical claim
— dropping iFeature would remove the comparison entirely; (3) the combined
ESM-2+SHAP50 experiment (§2.4.4) tests whether hand-crafted features add
incremental signal on top of the learned representation, which requires
iFeature features to exist.

### 4.6 Why all-33-layer probing over the initial 6-layer coarse sweep

The initial implementation probed 6 evenly-spaced layers (6/12/18/24/30/33)
as a cost-saving measure. This was revised to all 33 layers because: (a) the
GPU cost is unchanged — `repr_layers` accepts an arbitrary list and ESM-2
computes every layer's hidden state internally regardless of which are
returned, so requesting all 33 vs. 6 costs zero extra forward-pass time; (b) a
coarse 6-point sweep risks missing a genuine peak between sampled layers,
which matters for the "which biological information emerges at which depth"
question this experiment exists to answer; (c) a full-resolution curve is
what's typically shown in probing papers in this literature, and a paper
reviewer would likely ask for the missing layers if only 6 points were shown.

**Empirical confirmation (2026-07-01 run, NR variant):** the full sweep shows
ROC-AUC saturating at 1.0000 by layer 3 and staying there through layer 33 —
discriminability emerges almost immediately and does not improve further with
depth. ECE, however, keeps improving with depth even after ROC-AUC saturates:
0.0035 at layer 1 → 0.0015 at layer 3 → noisy in the 0.0005–0.0010 range
through the middle layers → 0.0002 at layer 33 (the best calibration value in
the entire study). This is the actual reportable finding from this experiment:
**separability and calibration quality are not the same signal and saturate
at different depths** — a 6-point coarse sweep would likely have missed the
noisy-middle / clean-end pattern in the ECE curve. See
`results/metrics/esm2_layer_table.csv` and
`results/figures/esm2_layer_probing.png` for the full sweep once §7 is filled
in from a complete run.

### 4.7 Why LR (not a non-linear model) as the standard classifier for the primary ESM-2 result

While the classifier battery (§2.3.2) evaluates 5 model families on ESM-2
features, LR is highlighted as the primary ESM-2 result specifically because
(a) it achieved the best calibration (ECE) in preliminary runs — the paper's
core added-value claim is about calibration, not just raw AUC, per Gap G2 in
§1.3; (b) a linear classifier on frozen embeddings is a much stronger claim
about representation quality than a non-linear one — if a linear probe
already separates the classes near-perfectly, that says the *embedding space*
itself has organized the biology, not that a flexible enough classifier can
paper over a messy embedding.

---

## 5. Known limitations (disclose in paper, do not hide)

1. **No exact-CV-fold reproduction of Raya et al.'s original 5-fold CV** —
   we use `random_state=42` throughout; the original paper's exact fold
   assignment is not published, so absolute numbers may differ slightly by
   chance even under otherwise identical methodology.
2. **iFeature descriptor coverage is Colab-version-dependent** — 19/22
   descriptors extracted (missing: KNN, ZScale, AAIndex = 389 of 18,899
   possible features, ≈2%). This affects the iFeature *baseline* arm only,
   not the ESM-2 primary result.
3. **No CD-HIT similarity-threshold deduplication** — only exact-duplicate
   removal is performed (§3.1). Near-duplicate sequences could still inflate
   CV performance on both ESM-2 and iFeature results equally, so this is
   unlikely to bias the *comparison* between the two, but could inflate both
   absolute numbers together.
4. **RecA negative set may be an easy contrast class** (Gap G5, inherited
   from the original paper, not fixed here) — RecA is cytoplasmic and
   functionally unrelated to EET; a harder negative set (outer-membrane
   proteins, flagellin) would be a stronger test of generalization. Out of
   scope for this pass.
5. **No independent held-out organism test** (Gap G6) — Raya et al. validated
   against 10 *O. alaskensis* G20 proteins not in training; we have not yet
   replicated this external validation step for either ESM-2 or iFeature.
6. **ESM-2 is used as a fixed feature extractor, not fine-tuned** — a
   fine-tuned ESM-2 (full or LoRA) might further improve results, but this
   requires materially more compute and is deliberately out of scope to keep
   the comparison to "off-the-shelf embeddings vs. hand-crafted features,"
   which is the more common and reproducible comparison in the cited
   literature (§1.4).
7. **Checkpoint-size ablation not performed** (§4.1) — only the 650M ESM-2
   variant was tested.

---

## 6. Reviewer checklist (for the AI coding agent / self-review pass)

Use this section as a literal checklist. For each item, verify against the
linked script and mark ✅ / ❌ / ⚠️ with a one-line note.

- [x] `04b_esm2_pipeline.py`: `PROBE_LAYERS` covers all 33 layers, not a subset
      — verified in the 2026-07-01 run (`esm2_layer_table.csv`, 33 rows)
- [x] `04b_esm2_pipeline.py`: `probe_layers()` reports bootstrap CI columns
      (`roc_auc_ci_low/high`, `pr_auc_ci_low/high`) in the output CSV
- [x] Best-layer selection breaks ties correctly — fixed 2026-07-01: with
      ROC-AUC saturated at 1.0000 for layers 3-33, the original
      `argmax(roc_auc)` picked layer 3 (first tie) rather than the
      best-calibrated layer. Now sorts by `(roc_auc, -ece)` so ties are
      broken by lowest ECE; layer 33 is correctly reported as best.
- [ ] `04b_esm2_pipeline.py` and `04c_esm2_analysis.py` use the **same**
      `StratifiedKFold` parameters (`n_splits=5, shuffle=True, random_state=42`)
- [ ] `04c_esm2_analysis.py`: iFeature rows are re-indexed to match ESM-2's
      `seq_id` order before comparison (label mismatch assertion present)
- [ ] `04c_esm2_analysis.py`: McNemar's test uses OOF (out-of-fold) predictions,
      not train-set or single-split predictions
- [ ] No data leakage: `StandardScaler` is inside each CV pipeline
      (fit only on training folds), not fit on the full dataset beforehand
- [ ] Bootstrap CI seed (`random_state=42`/`seed=42`) is consistent across
      `eval_utils.py`, `04b`, and `04c` for reproducibility
- [ ] All reported metrics in the paper draft trace back to a specific CSV
      file + column in `results/metrics/`, with no manually-typed numbers
- [ ] Every descriptor-count and feature-count claim in this document matches
      the actual `master_features.csv` shape from the latest Colab run
- [ ] Non-redundant vs. redundant variant is clearly labeled in every table
      (no silently mixing the two)

---

## 7. Results

> **Instructions:** fill in each subsection below by copying the relevant CSV
> contents (or a rendered table) directly from `results/metrics/` after the
> notebook run completes. Do not hand-type numbers — copy-paste from the CSV
> to avoid transcription errors. Include the exact CSV filename as a citation
> under each table.

### 7.1 Data sanity checks

*(source: `results/metrics/data_sanity.csv`)*

<!-- TODO: paste table -->

### 7.2 ESM-2 classifier battery (5-fold CV)

*(source: `results/metrics/esm2_cv_table.csv`)*

<!-- TODO: paste table -->

### 7.3 ESM-2 layer-wise probing (all 33 layers, with bootstrap 95% CI)

*(source: `results/metrics/esm2_layer_table.csv`, figure: `results/figures/esm2_layer_probing.png`)*

<!-- TODO: paste table + embed figure; note the best-performing layer and
whether its CI overlaps with layer 33's CI (i.e., is the "best" layer
statistically distinguishable from the default final layer) -->

### 7.4 ESM-2 embedding space visualization (UMAP)

*(source: `results/figures/esm2_umap.png`)*

<!-- TODO: embed figure + 1-2 sentence qualitative description of separability -->

### 7.5 iFeature baseline reproduction (Raya et al. 2025 Table 2)

*(source: `results/metrics/reproduction_table.csv`, `reproduction_table_nr.csv`)*

<!-- TODO: paste both NR and redundant tables side by side with paper's published numbers -->

### 7.6 iFeature extended pipeline (tuned + calibrated + stacked + SHAP)

*(source: `results/metrics/extended_table.csv`, `shap_top100.csv`, figures: `roc_curves.png`, `calibration_plots.png`, `shap_summary.png`)*

<!-- TODO: paste table + embed figures -->

### 7.7 ESM-2 vs. iFeature aligned comparison

*(source: `results/metrics/comparison_table.csv`, figure: `esm2_vs_ifeature_comparison.png`)*

<!-- TODO: paste table + embed figure -->

### 7.8 McNemar's test result

*(source: `results/metrics/mcnemar_result.csv`)*

<!-- TODO: paste table + 1-2 sentence interpretation -->

### 7.9 Bootstrap ROC-AUC difference CI

*(source: `results/metrics/auc_diff_ci.csv`)*

<!-- TODO: paste table + interpretation (does the CI exclude zero?) -->

### 7.10 Feature ablation study

*(source: `results/metrics/ablation_table.csv`, figure: `ablation_bar.png`)*

<!-- TODO: paste table + embed figure; note top 3 descriptor groups by ROC-AUC -->

### 7.11 Autoresearch loop log

*(source: `results/metrics/autoresearch_log.csv`)*

<!-- TODO: paste table; note which experiments were kept vs. reverted, and the final best PR-AUC -->

### 7.12 Headline summary table (for abstract / discussion)

<!-- TODO: one small table combining: best iFeature model, best ESM-2 model,
McNemar p-value, AUC-diff CI, best probing layer — this is the table the
paper's abstract will quote from -->

---

## 8. Open questions for reviewers

1. Is the exact-duplicate-only deduplication (no CD-HIT) an acceptable
   limitation to disclose, or should CD-HIT at 90% identity be run before
   these results are considered publication-ready? (§3.1, §7 limitation #3)
2. Does the McNemar's test + bootstrap-AUC-CI combination adequately support
   a claim of "ESM-2 significantly outperforms iFeature," or is a third test
   (e.g., a permutation test on the paired AUCs) warranted before that claim
   appears in the abstract?
3. Should a checkpoint-size ablation (ESM-2 35M vs 150M vs 650M) be run before
   submission, given that "650M is the standard choice in the cited
   literature" is currently asserted rather than empirically shown in this
   repo? (§4.1)
4. Is probing with LR only sufficient, or should a shallow MLP probe be added
   as a secondary check that the "best layer" finding isn't an artifact of
   LR's linearity specifically?
5. Given RecA is acknowledged as an "easy" negative class (Gap G5, inherited
   from the original paper), does the near-perfect ROC-AUC (both iFeature and
   ESM-2) mean this dataset is too easy to be a strong publication claim on
   its own, independent of the calibration finding?

---

## References

1. Raya D, Peta V, Bomgni A, Do TD, Kalimuthu J, Salem DR, Gadhamshetty V,
   Gnimpieba EZ, Dhiman SS. *Prediction and validation of nanowire proteins in
   Oleidesulfovibrio alaskensis G20 using machine learning and feature
   engineering.* Comput Struct Biotechnol J. 2025;27:1706–1718.
   doi:10.1016/j.csbj.2025.04.022
2. Chen Z, Zhao P, Li F, et al. iFeature: a Python package and web server for
   features extraction and selection from protein and peptide sequences.
   *Bioinformatics*. 2018;34(14):2499–2502.
3. Lin Z, Akin H, Rao R, et al. Evolutionary-scale prediction of atomic-level
   protein structure with a language model. *Science*. 2023;379(6637):1123-1130.
   (ESM-2 model paper)
4. Mall R, et al. Benchmarking protein language models for protein
   crystallization. *Sci Rep*. 2024.
5. Chu H, et al. Comprehensive Research on Druggable Proteins: From PSSM to
   Pre-Trained Language Models. *Int J Mol Sci*. 2024.
6. Du Z, et al. pLM4ACE: A protein language model based predictor for
   antihypertensive peptide screening. *Food Chemistry*. 2023.
7. Dietterich TG. Approximate statistical tests for comparing supervised
   classification learning algorithms. *Neural Computation*. 1998;10(7):1895-1923.
   (McNemar's test justification for classifier comparison)
