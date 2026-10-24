"""Compare baselines and course models using purged expanding-time validation.

Run after generate_data.py and feature_engineering.py. Final test results are
descriptive evidence for this synthetic experiment, not business impact.
"""
from pathlib import Path
import json
import platform
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix, accuracy_score)
from model import FEATURES, TARGET

ROOT = Path(__file__).resolve().parent

def pipeline(estimator):
    numeric = [c for c in FEATURES if c != 'category']
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), numeric),
        ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                               ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), ['category'])])
    return Pipeline([('preprocessing', preprocessing), ('model', estimator)])

def evaluate(y, score, threshold):
    pred = np.asarray(score) >= threshold
    return {'accuracy': float(accuracy_score(y, pred)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'average_precision': float(average_precision_score(y, score)),
            'roc_auc': float(roc_auc_score(y, score)),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            'alerts': int(pred.sum())}

def main():
    df = pd.read_csv(ROOT/'model_features.csv', parse_dates=['date']).sort_values('date')
    dates = sorted(df.date.unique())
    test_start = pd.Timestamp(dates[int(len(dates)*.8)])
    development = df[df.date < test_start-pd.Timedelta(days=7)]
    test = df[df.date >= test_start]
    dd = sorted(development.date.unique())
    folds = []
    for lo, hi in [(0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        start = pd.Timestamp(dd[int(len(dd)*lo)])
        end = pd.Timestamp(dd[min(int(len(dd)*hi),len(dd)-1)])
        train = development[development.date < start-pd.Timedelta(days=7)]
        val = development[(development.date >= start) & (development.date < end if hi<1 else development.date<=end)]
        assert train.date.max()+pd.Timedelta(days=7) < val.date.min()
        if train[TARGET].nunique()!=2 or val[TARGET].nunique()!=2:
            raise ValueError('Both classes required in each temporal fold')
        folds.append((train, val))
    models = {
        'MajorityBaseline': DummyClassifier(strategy='most_frequent'),
        'DecisionTree_leaf30': DecisionTreeClassifier(max_depth=5,min_samples_leaf=30,class_weight='balanced',random_state=42),
        'DecisionTree_leaf60': DecisionTreeClassifier(max_depth=5,min_samples_leaf=60,class_weight='balanced',random_state=42),
        'LogisticRegression': LogisticRegression(class_weight='balanced',max_iter=2000,random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=150,max_depth=7,min_samples_leaf=30,class_weight='balanced',random_state=42,n_jobs=-1)}
    rows, logs = [], []
    for name, estimator in models.items():
        truth, scores = [], []
        for i,(train,val) in enumerate(folds):
            pipe = pipeline(estimator)
            pipe.fit(train[FEATURES], train[TARGET])
            p = pipe.predict_proba(val[FEATURES])[:,1]
            truth.extend(val[TARGET].tolist()); scores.extend(p.tolist())
            logs.append({'model':name,'fold':i+1,'train_end':str(train.date.max().date()),'validation_start':str(val.date.min().date()),'validation_end':str(val.date.max().date()),'train_rows':len(train),'validation_rows':len(val),'average_precision':float(average_precision_score(val[TARGET],p))})
        truth, scores = np.array(truth), np.array(scores)
        thresholds = [.5] if name=='MajorityBaseline' else np.linspace(.05,.95,19)
        candidates = [(float(t),evaluate(truth,scores,t)) for t in thresholds]
        feasible = [(t,m) for t,m in candidates if m['recall'] >= .75]
        t,m = max(feasible,key=lambda r:(r[1]['precision'],r[1]['f1'])) if feasible else max(candidates,key=lambda r:r[1]['f1'])
        rows.append({'model':name,'threshold':t,'recall_goal_met':bool(feasible),**{k:v for k,v in m.items() if k!='confusion_matrix'}})
    table=pd.DataFrame(rows)
    # Select entirely on development validation, before inspecting any test score.
    eligible=table[table.model!='MajorityBaseline']
    winner=eligible.sort_values(['average_precision','f1'],ascending=False).iloc[0]
    selected=str(winner['model']); threshold=float(winner['threshold'])
    final={}
    for name, estimator in models.items():
        pipe=pipeline(estimator); pipe.fit(development[FEATURES],development[TARGET])
        t=float(table.loc[table.model==name,'threshold'].iloc[0])
        final[name]=evaluate(test[TARGET],pipe.predict_proba(test[FEATURES])[:,1],t)
        if name==selected:
            joblib.dump({'pipeline':pipe,'threshold':threshold,'features':FEATURES,'target':TARGET},ROOT/'comparison_selected_model.joblib')
    # Transparent operational baseline, using only snapshot-time information.
    rule=(test.days_of_supply < test.base_lead_time_days).astype(float)
    final['StockCoverageRule']=evaluate(test[TARGET],rule,.5)
    table.to_csv(ROOT/'model_comparison.csv',index=False)
    pd.DataFrame(logs).to_csv(ROOT/'temporal_validation.csv',index=False)
    result={'data':'Seeded synthetic FMCG simulation; no measured company savings',
        'selection_metric':'Pooled expanding-window validation average precision',
        'selected_model':selected,'selected_threshold':threshold,
        'test_start':str(test_start.date()),'development_rows':len(development),'test_rows':len(test),
        'test_positive_rate':float(test[TARGET].mean()),'test_results':final,
        'note':'Original model.py artifacts remain separate. Dashboard uses its original Decision Tree.',
        'environment':{'python':platform.python_version(),**{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','joblib']}}}
    (ROOT/'comparison_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
