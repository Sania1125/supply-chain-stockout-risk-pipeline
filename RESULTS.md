# Measured results

Seeded synthetic FMCG experiment; test observations: 1,600; positives: 79 (4.94%).

| Comparison candidate | Precision | Recall | F1 | Average precision | Alerts |
|---|---:|---:|---:|---:|---:|
| MajorityBaseline | 0.000 | 0.000 | 0.000 | 0.049 | 0 |
| DecisionTree_leaf30 | 0.275 | 0.481 | 0.350 | 0.330 | 138 |
| DecisionTree_leaf60 | 0.256 | 0.759 | 0.383 | 0.377 | 234 |
| LogisticRegression | 0.435 | 0.506 | 0.468 | 0.489 | 92 |
| RandomForest | 0.346 | 0.595 | 0.437 | 0.454 | 136 |
| StockCoverageRule | 0.140 | 0.924 | 0.244 | 0.133 | 520 |

Logistic Regression was selected by development validation average precision before test inspection. Its test recall is 0.506; it does not achieve the 75% validation recall objective on test. Do not change its threshold using these test results and claim another independent test. A simple stock-coverage rule captures more events but creates 520 alerts. Model choice depends on workload and missed-event costs.

The separately retained dashboard Decision Tree (model.py, earlier 60/20/20-style split) has test precision 0.251, recall 0.785, F1 0.380: TP=62, FN=17, FP=185, TN=1336. It uses a different training/selection protocol from compare_models.py and should not be treated as a controlled winner comparison. These metrics supersede the old main-branch numbers only for this revised FMCG dataset.

No real stockout reduction, money saved or company adoption is established.
