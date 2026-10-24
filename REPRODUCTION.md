# Original reproduction

Base repository commit: 84c50fed6ed0c8e2cff1e54a99cae1c9bb21abfa.
On 2026-10-24, ran the original model.py on the original committed model_features.csv before copying revised data. The rerun matched reported rounded metrics: precision 0.698, recall 0.508, F1 0.588, ROC-AUC 0.872, average precision 0.653. The confusion matrix matched [[1089,92],[206,213]]. This reproduces model fitting from committed features; the original generator was not successfully reproduced end-to-end, because its paths disagree with the README.

The revised generator changes product categories and seasonality as well as paths. It creates a new experiment, not a claim of higher accuracy on identical original data. Original versus revised results must not be used to attribute improvement to an algorithm alone.

Revised run: generate_data.py → feature_engineering.py → model.py → compare_models.py. comparison_results.json records package versions; temporal_validation.csv records folds. Git history identifies this revision. Submit the final branch commit using git rev-parse HEAD.
