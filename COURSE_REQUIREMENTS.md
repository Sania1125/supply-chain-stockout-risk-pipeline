# Requirements traceability

| Course requirement | Evidence | Status |
|---|---|---|
| Business decision, stakeholder, observation, target, horizon | PROJECT_REPORT.md | Defined |
| Data availability and provenance | generate_data.py; data dictionary | Synthetic only; no real-company claim |
| Initial Decision Tree | model.py; decision_rules.txt | Executed |
| Leakage prevention | feature_engineering.py; temporal_validation.csv | Temporal gaps; training-only preprocessing in comparison |
| Baseline and multiple models | compare_models.py; model_comparison.csv | Executed |
| Hyperparameter/threshold validation | model.py and compare_models.py | Limited grids; no test-driven tuning |
| EDA notebook | EDA.ipynb | Added; summary executed separately |
| Reproduce original result | REPRODUCTION.md | Rerun on preserved original feature CSV |
| Critique and improvements | PROJECT_REPORT.md; EXPERIMENT_LOG.md | Recorded |
| Explainability | decision_rules.txt; feature_importance.csv | Exported |
| Prediction-to-action mapping | PROJECT_REPORT.md; app.py | Manager review; no automatic purchasing |
| Dashboard/MVP | app.py | Existing code; live UI not retested here |
| Business KPIs | PROJECT_REPORT.md | Defined; impact unmeasured |
| AI usage log | AI_USAGE_LOG.md | Records this assistant revision only |
| Environment and run commands | README.md; comparison_results.json | Recorded |
| Exact final commit | Git branch head / git rev-parse HEAD | Obtain from submitted branch |
| Final slides and viva defense | Not included in this code revision | Remaining presentation task |

Sources: FMCG Inventory Intelligence Project Assignment (55 slides), ML Project Selection Audit Protocol (Word), MATH461 GitHub First ML Project Lecture (41 slides), ML Terminology and Evaluation (71 slides), Python data science stack (26 slides), Right Frame of Mind (43 slides), fundamentals lecture 2 (45 slides), From Business Problems (48 slides), Commercially Useful ML Projects (4 pages).

Mandatory gates are self-audited, not instructor approval. Data feasibility passes only for a synthetic educational prototype; real deployment remains conditional on actual data.
