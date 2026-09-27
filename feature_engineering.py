"""Build SKU × warehouse × day snapshots and a strictly future 7-day label."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
HORIZON = 7

def build():
    sales = pd.read_csv(ROOT / 'daily_sales.csv', parse_dates=['date'])
    inv = pd.read_csv(ROOT / 'inventory_snapshots.csv', parse_dates=['date'])
    products = pd.read_csv(ROOT / 'products.csv')
    suppliers = pd.read_csv(ROOT / 'suppliers.csv')
    keys = ['date', 'sku', 'warehouse_id']
    for name, frame in [('sales', sales), ('inventory', inv)]:
        if frame.duplicated(keys).any():
            raise ValueError(f'Duplicate {name} SKU × warehouse × day records')
    df = sales.merge(inv, on=keys, validate='one_to_one').merge(products, on='sku', validate='many_to_one').merge(suppliers, on='supplier_id', validate='many_to_one')
    df = df.sort_values(['sku', 'warehouse_id', 'date']).reset_index(drop=True)
    groups = df.groupby(['sku', 'warehouse_id'], sort=False)
    parts = []
    latest = []
    for _, g in groups:
        g = g.copy().reset_index(drop=True)
        g['avg_sales_7d'] = g.units_sold.rolling(7, min_periods=7).mean()
        g['avg_sales_30d'] = g.units_sold.rolling(30, min_periods=30).mean()
        g['demand_volatility_30d'] = g.units_sold.rolling(30, min_periods=30).std()
        g['days_of_supply'] = g.on_hand_qty / g.avg_sales_30d.clip(lower=0.01)
        g['month'] = g.date.dt.month
        g['day_of_week'] = g.date.dt.dayofweek
        # Shift first: the snapshot day's stockout never enters the label.
        future = pd.concat([g.stockout_flag.shift(-j) for j in range(1, HORIZON + 1)], axis=1)
        g['stockout_within_7d'] = future.max(axis=1)
        valid = g.iloc[29:-HORIZON:7].copy()
        parts.append(valid)
        latest.append(g.iloc[[-1]])
    columns = ['date','sku','warehouse_id','category','on_hand_qty','on_order_qty','avg_sales_7d','avg_sales_30d','demand_volatility_30d','days_of_supply','base_lead_time_days','reliability_score','unit_cost','month','day_of_week']
    features = pd.concat(parts, ignore_index=True)[columns + ['stockout_within_7d']]
    features['stockout_within_7d'] = features.stockout_within_7d.astype(int)
    live = pd.concat(latest, ignore_index=True)[columns + ['product_name','supplier_id']]
    if features.isna().any().any() or live.isna().any().any():
        raise ValueError('Missing feature values')
    features.to_csv(ROOT / 'model_features.csv', index=False)
    live.to_csv(ROOT / 'latest_snapshot_features.csv', index=False)
    print(f'Historical rows: {len(features)}; positive rate: {features.stockout_within_7d.mean():.3f}; latest rows: {len(live)}')

if __name__ == '__main__':
    build()
