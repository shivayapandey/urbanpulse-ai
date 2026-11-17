from pathlib import Path
import hashlib,json,duckdb,pandas as pd,numpy as np,joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
R=Path(__file__).resolve().parents[1];P=R/'data/processed';q=json.loads((P/'quality.json').read_text());m=json.loads((P/'metrics.json').read_text());bt=json.loads((P/'backtests.json').read_text());c=duckdb.connect(str(P/'urbanpulse.duckdb'),read_only=True)
for f in q['source_files']:
 p=R/'data/raw'/f['name'];assert p.stat().st_size==f['bytes']; assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
raw=c.execute('select count(*) from raw').fetchone()[0]; clean=c.execute('select count(*) from trips').fetchone()[0]; hz=c.execute('select sum(trips) from hourly_zone').fetchone()[0];flow=c.execute('select sum(trips) from flows').fetchone()[0]
assert (raw,clean,hz,flow)==(q['n'],q['cleaned'],q['cleaned'],q['cleaned']);print('raw, clean, excluded, hourly, flow:',raw,clean,raw-clean,hz,flow)
assert c.execute("select min(pickup),max(pickup) from trips").fetchone()[0].year==2025
# Independently regenerate features and predictions; actual previous test observations are used at each one-step origin.
y=c.execute('select "hour",sum(trips) trips from hourly_zone group by 1 order by 1').df().set_index('hour').asfreq('h',fill_value=0)
for lag in [1,24,168]:y[f'lag_{lag}']=y.trips.shift(lag)
y['rolling_24']=y.trips.shift(1).rolling(24).mean();y['hour']=y.index.hour;y['weekday']=y.index.dayofweek;y=y.dropna();features=m['features'];model=joblib.load(P/'forecast_model.joblib');pred=y.loc['2025-03-01':];saved=pd.read_parquet(P/'forecast.parquet').set_index('hour')
assert len(pred)==len(saved)==m['test_n'];assert np.allclose(saved.model,model.predict(pred[features]));assert np.allclose(saved.trips,pred.trips);assert np.allclose(saved.seasonal_naive,pred.lag_168)
for col in ['model','seasonal_naive']:
 mae=mean_absolute_error(saved.trips,saved[col]);rmse=np.sqrt(mean_squared_error(saved.trips,saved[col]));assert round(mae,2)==m[col]['MAE'] and round(rmse,2)==m[col]['RMSE'];print(col,round(mae,2),round(rmse,2))
for x in bt:
 start=pd.Timestamp(x['test_start']); end=pd.Timestamp(x['test_end']);tr=y.index<start;te=(y.index>=start)&(y.index<=end);est=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_leaf_nodes=15,l2_regularization=2,random_state=42).fit(y.loc[tr,features],y.loc[tr,'trips']);a=y.loc[te,'trips'];pr=est.predict(y.loc[te,features]);base=y.loc[te,'lag_168'];assert len(a)==x['n'] and round(mean_absolute_error(a,pr),2)==x['model_mae'] and round(mean_absolute_error(a,base),2)==x['baseline_mae']; assert round(np.sqrt(mean_squared_error(a,pr)),2)==x['model_rmse']; assert round(np.sqrt(mean_squared_error(a,base)),2)==x['baseline_rmse'];print('backtest',str(start.date()),x['model_mae'],x['baseline_mae'])
print('improvement MAE %.2f%% RMSE %.2f%%'%((1-m['model']['MAE']/m['seasonal_naive']['MAE'])*100,(1-m['model']['RMSE']/m['seasonal_naive']['RMSE'])*100))
# Refit the March model from the documented training period and compare predictions.
train=y.index<'2025-03-01'; fresh=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_leaf_nodes=15,l2_regularization=2,random_state=42).fit(y.loc[train,features],y.loc[train,'trips']);assert np.allclose(fresh.predict(pred[features]),saved.model);print('March model refit reproducible')
rule_sql={
'missing_required':"tpep_pickup_datetime IS NULL OR tpep_dropoff_datetime IS NULL OR PULocationID IS NULL OR DOLocationID IS NULL OR trip_distance IS NULL",
'pickup_outside_period':"tpep_pickup_datetime < TIMESTAMP '2025-01-01' OR tpep_pickup_datetime >= TIMESTAMP '2025-04-01'",
'invalid_duration':"tpep_dropoff_datetime <= tpep_pickup_datetime OR date_diff('second',tpep_pickup_datetime,tpep_dropoff_datetime)>14400",
'invalid_distance':"trip_distance <= 0 OR trip_distance > 100",
'invalid_location':"PULocationID NOT BETWEEN 1 AND 263 OR DOLocationID NOT BETWEEN 1 AND 263 OR PULocationID NOT IN (SELECT LocationID FROM zones WHERE Borough NOT IN ('Unknown','EWR')) OR DOLocationID NOT IN (SELECT LocationID FROM zones WHERE Borough NOT IN ('Unknown','EWR'))"}
for key,condition in rule_sql.items():
 n=c.execute(f'SELECT count(*) FROM raw WHERE {condition}').fetchone()[0];assert n==q[key],(key,n,q[key]);print('quality',key,n)
assert c.execute('select cast("hour" as date),sum(trips) from hourly_zone group by 1 order by 1').fetchall()==c.execute('select trip_day,sum(trips) from flows group by 1 order by 1').fetchall();print('Daily hourly/flow totals reconcile')
