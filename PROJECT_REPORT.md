# Supply Chain & Stockout Risk Intelligence Platform

## 1. Business problem, stakeholder and decision
An FMCG distributor or warehouse inventory manager needs to decide which product-location combinations require replenishment review today. Shortages can lose sales; excessive orders tie up working capital. This student prototype ranks next-seven-day inventory-depletion risk. No real Multan company has supplied or validated the current data.

Primary task: supervised binary classification. Demand forecasting, automatic order optimization, expiry prediction and supplier-delay prediction are extensions, not implemented predictive functions. A Streamlit what-if simulation is separate from the fitted risk model.

Observation: one SKU × warehouse × end-of-day snapshot. Inputs are available after that day's sales and inventory snapshot. Target: 1 if any of the following seven days ends with zero stock, otherwise 0. This is an operational stock-depletion proxy: zero stock does not always prove unfulfilled demand. Simulated units_demanded minus units_sold can measure lost-sales units separately.

## 2. Data source and dictionary
The revised generator uses seeded synthetic data: 40 FMCG SKUs, four generic warehouses, ten fictional suppliers, 365 days in 2023. Seasonal factors and costs are illustrative; neither Pakistan-specific estimates nor verified market prices. Daily sales and inventory each contain 58,400 records. Historical features contain 7,520 snapshots sampled weekly after a 30-day warm-up and excluding the final seven unlabeled days. The latest snapshot has 160 rows.

| Variable | Meaning | Role |
|---|---|---|
| date, sku, warehouse_id | End-of-day snapshot identity | Keys; not direct model inputs |
| category | FMCG product category | Categorical input |
| on_hand_qty | Remaining usable units after daily sales | Numeric input |
| on_order_qty | Open-order units at inventory snapshot | Numeric input |
| avg_sales_7d, avg_sales_30d | Trailing mean fulfilled sales, including snapshot day | Numeric inputs |
| demand_volatility_30d | Trailing standard deviation of fulfilled sales | Numeric input; sales variability proxy |
| days_of_supply | Stock / max(30-day average sales, 0.01) | Engineered input; denominator floor avoids zero division |
| base_lead_time_days | Supplier's simulated baseline lead time | Numeric input |
| reliability_score | Fixed simulated supplier characteristic | Numeric input; not estimated from future outcomes |
| unit_cost | Illustrative synthetic unit cost | Numeric input |
| month, day_of_week | Snapshot calendar information | Numeric inputs |
| stockout_within_7d | Any zero-stock flag on following days 1–7 | Binary target only |
| units_demanded | Simulated latent demand | Analysis only, excluded from predictors |
| actual_delivery_date | Eventual simulated delivery outcome | Excluded from predictors |

Sales are capped by stock. Therefore trailing sales may underestimate underlying demand during shortages. The generator records inventory before placing a new order at the end of the day; on_order_qty follows that convention. Business deployment must define and reconcile this timing explicitly.

## 3. Pipeline and methodology
Business problem → synthetic operational records → joins and checks → trailing features and future labels → chronological splits → baseline and models → temporal validation → interpretation → risk review queue → measured operational outcomes.

feature_engineering.py rejects duplicate daily keys, validates joins, computes trailing features within each SKU-warehouse, and rejects missing model values. It does not currently repair arbitrary dirty business files or verify a complete daily calendar. Real imports will need those checks.

model.py retains the initial course Decision Tree workflow: depth 5, balanced class weights, leaf-size and threshold tuning on validation only, chronological training/validation/test with a full horizon gap. Feature categories are aligned to training columns. Trees do not need standardization. Exported rules describe tree splits; alert thresholds may differ from the tree's default class decision.

compare_models.py adds a majority baseline, a stock-coverage rule, Logistic Regression, two Decision Tree configurations and Random Forest. Imputation, one-hot encoding and scaling are fitted inside each training fold. Three expanding-window folds exclude any training label whose seven-day outcome reaches validation. The final chronological test period is held out from selection. Model-family selection uses pooled validation average precision; thresholds target 75% validation recall, maximizing precision among feasible choices. If the goal is infeasible, the script explicitly records that and chooses validation F1 instead.

The comparison-selected model is stored separately; the dashboard continues to use model.py's interpretable Decision Tree. Comparison does not silently replace the deployed model. Class-weighted model scores are not guaranteed calibrated probabilities; probability calibration remains future work.

## 4. Evaluation and business meaning
See RESULTS.md and comparison_results.json for measured synthetic results. Accuracy alone is misleading because the test positive rate is low. Report precision, recall, F1, average precision, ROC-AUC, confusion matrix and alert workload. average_precision_score is average precision, a stepwise summary of the PR curve, not trapezoidal PR-AUC. The legacy model.py JSON names that quantity pr_auc.

A false negative misses a future stock-depletion event; a false positive creates unnecessary review. Threshold selection must ultimately consider staff capacity, lost-sales costs and holding costs. Higher recall can require many more reviews.

High: review current stock and open POs today; consider expediting, transferring stock or replenishment. Medium: monitor and review tomorrow. Low: routine monitoring. These are review priorities, not an automatic order quantity or purchase order. The seven-day horizon can be shorter than supplier lead time; alternative actions or a longer-horizon model may be needed.

Business KPIs: stockout-day rate, unit fill rate (fulfilled demand / requested demand), lost-sales units, review workload and holding cost. No actual savings or causal policy improvement has been measured. A future simulation must compare policies under the same demand/supply realizations, costs and constraints.

## 5. Course concepts applied
Supervised learning maps snapshot features to observed future labels. Feature engineering summarizes sales and stock coverage. Decision Trees learn interpretable nonlinear rules. Class weights address rare positives. Depth and minimum leaf size control overfitting. Expanding-window validation tests future-period generalization. Training-only preprocessing prevents contamination. Baselines reveal whether added complexity improves decisions. Thresholds translate scores into review workload. Explainability helps managers inspect the fitted relationship but does not establish causation.

## 6. Progress, ownership and limitations
Implemented and executed in this revision: reproducible FMCG generator, existing feature and Decision Tree pipelines, temporal model comparison, metrics, data dictionary and EDA notebook. Existing dashboard code remains; a live browser demonstration has not been verified in this revision.

Original repository state is preserved on main and by the recorded base commit in REPRODUCTION.md. The original model was rerun before changing the experiment. The revised FMCG data is a separate experiment, so before/after metric changes are not evidence of algorithm improvement on identical data. Controlled comparisons among revised candidates use the same revised folds and test set.

Remaining: real business data access, calendar/timing validation, calibration, error analysis across warehouses/categories, demand-censoring treatment, independent future test periods, staff review and policy-impact measurement. PCA, clustering, GMM and KDE are optional extensions only when a concrete inventory decision justifies them. They are not required merely to increase algorithm count.
