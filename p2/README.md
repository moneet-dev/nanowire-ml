# P2 — Biofilm Knowledge Graph + Graph-ML Link Prediction

**Coming soon.**

Placeholder for the second project: building a CREID protein–protein interaction
knowledge graph and running an autoresearch loop over a GraphSAGE / GAE link
predictor (benchmark: link-prediction AUC / Hits@10 on held-out edges).

This directory is intentionally reserved so P2 notebooks and modules slot in
alongside P1 without restructuring the repo. The following P1 components are
designed for direct reuse here:

- `src/data_utils.py` — FASTA loading accepts any sequence file (CREID proteins).
- `src/feature_utils.py` — `extract_all_features()` produces node features for GNNs.
- `src/eval_utils.py` — ECE / calibration utilities transfer to edge-pair scoring.
- `src/classify.py` — the `### BEGIN/END EXPERIMENT ###` autoresearch interface
  is reused by a future `gnn_train.py` with the same keep/revert loop.
