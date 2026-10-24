# Experiment log — 2026-10-24

| Observed weakness | Hypothesis/change | Experiment | Finding |
|---|---|---|---|
| Generator writes data/raw while feature script reads root | Resolve paths relative to script | Run generator and feature script | 58,400 daily records; 7,520 historical rows |
| Non-FMCG categories | Use six explicitly synthetic FMCG categories and illustrative seasonal factors | Regenerate complete data | New dataset; results not directly comparable to old dataset |
| Missing baseline/model-family evidence | Compare majority/rule baselines, DT, LR and RF on common temporal partitions | Three expanding validation windows and final test | See model_comparison.csv and RESULTS.md |
| High validation recall may not generalize | Evaluate frozen thresholds on later period | Held-out test | Selected LR recall 0.506 despite validation target; shift requires investigation |

Provenance: base commit recorded in REPRODUCTION.md; this branch contains the complete revised source and generated outputs. No hyperparameter changes were made after viewing test results.
