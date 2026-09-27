"""Decision Tree for next-7-day stockout risk; time-separated tuning and test."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent
TARGET = 'stockout_within_7d'
FEATURES = ['on_hand_qty','on_order_qty','avg_sales_7d','avg_sales_30d','demand_volatility_30d','days_of_supply','base_lead_time_days','reliability_score','unit_cost','month','day_of_week','category']

def encode(frame, columns=None):
    x = pd.get_dummies(frame[FEATURES], columns=['category'], dtype=int)
    return x if columns is None else x.reindex(columns=columns, fill_value=0)

def metrics(y, probability, threshold):
    pred = (probability >= threshold).astype(int)
    return {'precision': round(float(precision_score(y,pred,zero_division=0)),3), 'recall': round(float(recall_score(y,pred,zero_division=0)),3), 'f1': round(float(f1_score(y,pred,zero_division=0)),3), 'roc_auc': round(float(roc_auc_score(y,probability)),3), 'pr_auc': round(float(average_precision_score(y,probability)),3), 'confusion_matrix': confusion_matrix(y,pred,labels=[0,1]).tolist()}

def main():
    df = pd.read_csv(ROOT / 'model_features.csv', parse_dates=['date']).sort_values('date')
    dates = sorted(df.date.unique())
    # A full horizon separates labels in each earlier partition from later features.
    first = dates[int(len(dates)*0.60)]
    second = dates[int(len(dates)*0.80)]
    gap = pd.Timedelta(days=7)
    train = df[df.date < first-gap]
    validation = df[(df.date >= first) & (df.date < second-gap)]
    test = df[df.date >= second]
    if any(part[TARGET].nunique() != 2 for part in (train,validation,test)):
        raise ValueError('Each time partition needs both target classes')
    x_train = encode(train)
    cols = x_train.columns.tolist()
    x_val, x_test = encode(validation, cols), encode(test, cols)
    # Choose complexity and alert threshold on validation only. Test remains untouched.
    candidates = []
    for leaf in [10,30,60]:
        tree = DecisionTreeClassifier(max_depth=5,min_samples_leaf=leaf,class_weight='balanced',random_state=42)
        tree.fit(x_train, train[TARGET])
        p = tree.predict_proba(x_val)[:,1]
        for threshold in np.arange(0.20,0.81,0.05):
            score = metrics(validation[TARGET],p,threshold)
            if score['recall'] >= 0.75:
                candidates.append((score['precision'],score['f1'],leaf,round(float(threshold),2),tree))
    if not candidates:
        raise ValueError('No validation configuration meets 75% recall; review decision policy')
    _,_,leaf,threshold,model = max(candidates,key=lambda row: row[:2])
    test_p = model.predict_proba(x_test)[:,1]
    report = {'model':'DecisionTreeClassifier','target':TARGET,'prediction_horizon_days':7,'observation_unit':'SKU × warehouse × snapshot day','split':{'train_end':str(train.date.max().date()),'validation_start':str(validation.date.min().date()),'validation_end':str(validation.date.max().date()),'test_start':str(test.date.min().date()),'rows':{'train':len(train),'validation':len(validation),'test':len(test)}},'selection':{'min_samples_leaf':leaf,'max_depth':5,'threshold':threshold,'validation':metrics(validation[TARGET],model.predict_proba(x_val)[:,1],threshold)},'test':metrics(test[TARGET],test_p,threshold)}
    (ROOT/'model_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.DataFrame({'feature':cols,'importance':model.feature_importances_}).sort_values('importance',ascending=False).to_csv(ROOT/'feature_importance.csv',index=False)
    (ROOT/'decision_rules.txt').write_text(export_text(model,feature_names=cols,max_depth=5))
    joblib.dump({'model':model,'feature_cols':cols,'threshold':threshold,'target':TARGET},ROOT/'stockout_model.joblib')
    live = pd.read_csv(ROOT/'latest_snapshot_features.csv')
    p = model.predict_proba(encode(live,cols))[:,1]
    live['stockout_risk_score'] = np.round(p,3)
    live['risk_tier'] = np.where(p>=threshold,'High',np.where(p>=threshold/2,'Medium','Low'))
    live['recommended_action'] = np.where(p>=threshold,'Review stock and open POs; consider replenishment today',np.where(p>=threshold/2,'Monitor stock and review tomorrow','Continue routine monitoring'))
    live.to_csv(ROOT/'model_output.csv',index=False)
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
