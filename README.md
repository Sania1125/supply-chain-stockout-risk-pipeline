# FMCG 7-day stockout risk decision support

This student project helps a **warehouse inventory manager** decide which SKU and warehouse combinations to review for replenishment. The observation is **one SKU × warehouse × snapshot day**. The Decision Tree predicts whether the SKU will have **any stockout on days 1–7 after that day** (`stockout_within_7d`: 1 = yes, 0 = no). A High alert means review stock and open purchase orders today, then decide whether and how much to order. The prediction does not automatically issue a PO.

Data is **synthetic**: 40 SKUs, 4 warehouses and one simulated year. Current stock, open order quantity, trailing sales, demand variation, supplier lead time and reliability, cost, and calendar features are available at snapshot time. The future stockout flag is used **only as a label**. All files are in the repository root.

## Reproduce

```bash
python -m pip install -r requirements.txt
python generate_data.py          # optional: recreate seeded synthetic raw CSVs
python feature_engineering.py   # model_features.csv; latest_snapshot_features.csv
python model.py                 # model_metrics.json; decision_rules.txt; model_output.csv
streamlit run app.py            # review queue and separate simulation
```

`generate_data.py` requires `faker` and writes the raw CSVs into the root. `load_data.py` can load raw CSVs into SQLite; SQL examples are in `schema.sql` and `queries.sql`. `model_output.csv` can also be imported into Power BI; see `build_guide.md` and `dax_measures.md` for the existing optional dashboard guidance. The Streamlit app has a separate what-if simulator; its controls do not affect the saved model.

## Method and checks

- Features use information up to and including the snapshot day. The label examines only the **next seven complete days**. The last seven days cannot be labeled and are excluded from training.
- Historical snapshots are sampled every seven days. Training, validation and held-out test are chronological, with at least seven days between partitions so an earlier label cannot reach a later partition.
- Decision Tree depth is limited to five. Minimum leaf size and alert threshold are chosen using validation data to target at least 75% recall, then evaluated once on the test period. `decision_rules.txt` and `feature_importance.csv` show interpretable rules and feature importance.
- `model_metrics.json` includes date ranges, row counts, threshold, confusion matrix, precision, recall, F1, ROC-AUC and PR-AUC. Current held-out test precision is **0.698**, recall **0.508**, F1 **0.588**, ROC-AUC **0.872**. Validation recall reached **0.789**, but the test recall fell below the goal. Do not claim a 75% test recall.
- A High alert triggers an immediate manager review; Medium triggers monitoring and review tomorrow; Low follows routine monitoring. Business value could be tracked by stockout days, lost-sales units, fill rate, and review workload. This synthetic data does **not** establish actual savings or deployment readiness.

The seven-day horizon, unit, stakeholder, decision, data source, leakage control, explainability and action mapping address the mandatory project audit gates. The feature pipeline cannot infer live purchase decisions or actual supplier changes. Before business use, obtain real dated transactions, validate data quality and drift, measure the cost of false alarms and missed stockouts, and test the alert policy with inventory staff.
