from pathlib import Path
import json, time, hashlib, logging, zipfile, joblib
import duckdb, pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data/raw'; OUT=ROOT/'data/processed'; OUT.mkdir(parents=True,exist_ok=True)
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
t0=time.perf_counter(); timings={}
def stage(name, fn):
    t=time.perf_counter(); result=fn(); timings[name]=round(time.perf_counter()-t,3); logging.info('%s: %.2fs',name,timings[name]); return result
files=sorted(RAW.glob('yellow_tripdata_2025-*.parquet'))
if len(files)!=3: raise RuntimeError('Expected January–March 2025 files in data/raw')
con=duckdb.connect(str(OUT/'urbanpulse.duckdb')); con.execute("SET threads=4")
con.execute("CREATE OR REPLACE VIEW raw AS SELECT * FROM read_parquet('"+str(RAW/'yellow_tripdata_2025-*.parquet')+"', union_by_name=true, filename=true)")
required=['tpep_pickup_datetime','tpep_dropoff_datetime','PULocationID','DOLocationID','trip_distance']
cols=[r[0] for r in con.execute('DESCRIBE raw').fetchall()]
if not set(required)<=set(cols): raise ValueError('Missing required schema fields')
con.execute("CREATE OR REPLACE TABLE zones AS SELECT * FROM read_csv_auto('"+str(RAW/'taxi_zone_lookup.csv')+"')")
# Independent rule counts overlap. Unknown/outside IDs 264/265 are excluded, as are zones without a mapped polygon.
rules={
 'missing_required':"tpep_pickup_datetime IS NULL OR tpep_dropoff_datetime IS NULL OR PULocationID IS NULL OR DOLocationID IS NULL OR trip_distance IS NULL",
 'pickup_outside_period':"tpep_pickup_datetime < TIMESTAMP '2025-01-01' OR tpep_pickup_datetime >= TIMESTAMP '2025-04-01'",
 'invalid_duration':"tpep_dropoff_datetime <= tpep_pickup_datetime OR date_diff('second',tpep_pickup_datetime,tpep_dropoff_datetime)>14400",
 'invalid_distance':"trip_distance <= 0 OR trip_distance > 100",
 'invalid_location':"PULocationID NOT BETWEEN 1 AND 263 OR DOLocationID NOT BETWEEN 1 AND 263 OR PULocationID NOT IN (SELECT LocationID FROM zones WHERE Borough NOT IN ('Unknown','EWR')) OR DOLocationID NOT IN (SELECT LocationID FROM zones WHERE Borough NOT IN ('Unknown','EWR'))"}
quality=stage('quality_scan',lambda: con.execute('SELECT count(*) n, '+', '.join(f'count(*) FILTER (WHERE {v}) AS {k}' for k,v in rules.items())+', '+', '.join(f'count(*) FILTER (WHERE "{c}" IS NULL) AS "null_{c}"' for c in required)+' FROM raw').df().iloc[0].to_dict())
condition=' AND '.join(f'NOT coalesce(({x}), true)' for x in rules.values())
def clean():
 con.execute(f"COPY (SELECT tpep_pickup_datetime pickup, tpep_dropoff_datetime dropoff, PULocationID pu, DOLocationID do_id, trip_distance distance, date_diff('second',tpep_pickup_datetime,tpep_dropoff_datetime)/60.0 duration FROM raw WHERE {condition}) TO '{OUT / 'trips.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)"); con.execute(f"CREATE OR REPLACE VIEW trips AS SELECT * FROM read_parquet('{OUT / 'trips.parquet'}')")
stage('clean_materialize',clean)
quality['cleaned']=con.execute('SELECT count(*) FROM trips').fetchone()[0]
# Exact duplicate rows over all published source fields; no automatic deletion without a trip identifier.
def dup():
 columns=', '.join(chr(34)+c+chr(34) for c in cols if c != 'filename'); return con.execute(f'SELECT coalesce(sum(n-1),0) FROM (SELECT count(*) n FROM raw GROUP BY {columns} HAVING count(*)>1)').fetchone()[0]
quality['exact_duplicate_excess_rows']=stage('exact_duplicate_scan',dup)
def aggregate():
 con.execute("CREATE OR REPLACE TABLE hourly_zone AS SELECT date_trunc('hour',pickup) AS \"hour\", pu AS zone_id, count(*) trips, avg(distance) mean_distance, avg(duration) mean_duration FROM trips GROUP BY 1,2")
 con.execute("CREATE OR REPLACE TABLE flows AS SELECT cast(pickup AS DATE) AS trip_day, hour(pickup) AS \"hour\", pu, do_id, count(*) trips FROM trips GROUP BY 1,2,3,4")
stage('aggregations_and_parquet',aggregate)
quality['coverage']=con.execute('SELECT min(pickup), max(pickup), count(distinct pu), count(distinct do_id), min(distance), max(distance), max(duration) FROM trips').fetchone()
quality['monthly']=con.execute("SELECT strftime(pickup,'%Y-%m') AS period, count(*) trips, avg(distance) mean_distance, avg(duration) mean_duration FROM trips GROUP BY 1 ORDER BY 1").df().to_dict('records')
# One-step ahead evaluation: the immediately previous hour has completed before each prediction.
y=con.execute("SELECT \"hour\", sum(trips) AS trips FROM hourly_zone GROUP BY 1 ORDER BY 1").df().set_index('hour').asfreq('h',fill_value=0)
for lag in [1,24,168]: y[f'lag_{lag}']=y.trips.shift(lag)
y['rolling_24']=y.trips.shift(1).rolling(24).mean(); y['hour']=y.index.hour; y['weekday']=y.index.dayofweek
y=y.dropna(); train=y.index < '2025-03-01'; test=~train
features=['lag_1','lag_24','lag_168','rolling_24','hour','weekday']
def model_run():
 model=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_leaf_nodes=15,l2_regularization=2,random_state=42)
 model.fit(y.loc[train,features],y.loc[train,'trips']); return model
model=stage('model_training',model_run)
joblib.dump(model, OUT/'forecast_model.joblib')
p=y.loc[test,['trips','lag_168']].copy(); p['model']=model.predict(y.loc[test,features]); p=p.rename(columns={'lag_168':'seasonal_naive'}); p.index.name='hour'; p.reset_index().to_parquet(OUT/'forecast.parquet',index=False)
metrics={k:{'MAE':round(mean_absolute_error(p.trips,p[k]),2),'RMSE':round(np.sqrt(mean_squared_error(p.trips,p[k])),2)} for k in ['seasonal_naive','model']}
metrics.update({'train_start':str(y.index[train].min()),'train_end':str(y.index[train].max()),'test_start':str(p.index.min()),'test_end':str(p.index.max()),'test_n':len(p),'features':features})
# Expanding training windows; each February test week is chronologically held out.
backtests=[]
for start in pd.to_datetime(['2025-02-01','2025-02-08','2025-02-15','2025-02-22']):
 end=start+pd.Timedelta(days=7); tr=y.index<start; te=(y.index>=start)&(y.index<end)
 estimator=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_leaf_nodes=15,l2_regularization=2,random_state=42)
 estimator.fit(y.loc[tr,features],y.loc[tr,'trips']); actual=y.loc[te,'trips']; prediction=estimator.predict(y.loc[te,features]); baseline=y.loc[te,'lag_168']
 backtests.append({'train_start':str(y.index[tr].min()),'train_end':str(y.index[tr].max()),'test_start':str(y.index[te].min()),'test_end':str(y.index[te].max()),'n':int(te.sum()),'baseline_mae':round(mean_absolute_error(actual,baseline),2),'baseline_rmse':round(np.sqrt(mean_squared_error(actual,baseline)),2),'model_mae':round(mean_absolute_error(actual,prediction),2),'model_rmse':round(np.sqrt(mean_squared_error(actual,prediction)),2)})
(OUT/'backtests.json').write_text(json.dumps(backtests,indent=2))
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2,default=str))
quality['timings_seconds']=timings; quality['source_files']=[{'name':f.name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files]; quality['total_elapsed_seconds']=round(time.perf_counter()-t0,2)
(OUT/'quality.json').write_text(json.dumps(quality,indent=2,default=str))
print(json.dumps({'quality':quality,'forecast':metrics},indent=2,default=str))
