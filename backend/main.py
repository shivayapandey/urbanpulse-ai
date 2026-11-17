from pathlib import Path
from functools import lru_cache
from datetime import date
import json, os, tempfile, zipfile
from threading import Lock
import duckdb, pandas as pd, numpy as np, shapefile
from pyproj import Transformer
from sklearn.cluster import KMeans
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'processed'; RAW=ROOT/'data'/'raw'
app=FastAPI(title='UrbanPulse API',version='1.0')
origins = ['http://127.0.0.1:5173', 'http://localhost:5173']
origins += [origin.strip().rstrip('/') for origin in os.getenv('CORS_ORIGINS', '').split(',') if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['GET'], allow_headers=['*'])
_db_lock = Lock()
def rows(sql,params=()):
 with _db_lock:
  with duckdb.connect(str(DATA/'urbanpulse.duckdb'), read_only=True) as c:
   c.execute("SET threads = 1")
   c.execute("SET memory_limit = '160MB'")
   return c.execute(sql, params).df()
def bounds(start,end):
 try: a=date.fromisoformat(start); b=date.fromisoformat(end)
 except ValueError: raise HTTPException(422,'Dates must use YYYY-MM-DD')
 if a>b or a<date(2025,1,1) or b>date(2025,3,31): raise HTTPException(422,'Select a date range within January–March 2025')
 return a,b
@lru_cache(maxsize=32)
def get_zone_data(start,end):
 a,b=bounds(start,end)
 return rows('SELECT zone_id, sum(trips) trips, sum(mean_duration*trips)/sum(trips) duration, sum(mean_distance*trips)/sum(trips) distance FROM hourly_zone WHERE CAST("hour" AS DATE) BETWEEN ? AND ? GROUP BY 1',[a,b])
@lru_cache(maxsize=1)
def geometry():
 with tempfile.TemporaryDirectory() as d:
  with zipfile.ZipFile(RAW/'taxi_zones.zip') as z:z.extractall(d)
  sf=shapefile.Reader(str(Path(d)/'taxi_zones'/'taxi_zones.shp')); fields=[f[0] for f in sf.fields[1:]]; trans=Transformer.from_crs(2263,4326,always_xy=True); feats=[]
  for r in sf.iterShapeRecords():
   rec=dict(zip(fields,r.record)); geo=r.shape.__geo_interface__
   def convert(v):
    if isinstance(v[0],(float,int)): return list(trans.transform(v[0],v[1]))
    return [convert(x) for x in v]
   feats.append({'type':'Feature','properties':{'id':int(rec['LocationID'])},'geometry':{'type':geo['type'],'coordinates':convert(geo['coordinates'])}})
  return {'type':'FeatureCollection','features':feats}
@lru_cache(maxsize=1)
def zones():return rows('SELECT LocationID AS id, Zone AS name, Borough AS borough FROM zones').to_dict('records')
@app.get('/api/health')
def health():return {'status':'ok','dataset':'NYC TLC yellow taxi','period':'2025-01 to 2025-03'}
@app.get('/api/overview')
def overview(start:str='2025-01-01',end:str='2025-03-31'):
 a,b=bounds(start,end); z=get_zone_data(start,end); names={x['id']:x['name'] for x in zones()}; t=rows('SELECT CAST("hour" AS DATE) AS day_date, sum(trips) trips FROM hourly_zone WHERE CAST("hour" AS DATE) BETWEEN ? AND ? GROUP BY 1 ORDER BY 1',[a,b]); hour=rows('SELECT hour("hour") h, sum(trips) trips FROM hourly_zone WHERE CAST("hour" AS DATE) BETWEEN ? AND ? GROUP BY 1 ORDER BY trips DESC LIMIT 1',[a,b]); total=int(z.trips.sum())
 return {'trips':total,'duration':float((z.duration*z.trips).sum()/total) if total else 0,'distance':float((z.distance*z.trips).sum()/total) if total else 0,'peak_hour':int(hour.iloc[0].h) if len(hour) else None,'top_zone':names.get(int(z.sort_values('trips',ascending=False).iloc[0].zone_id),'Unknown') if len(z) else None,'daily':t.assign(day=t.day_date.astype(str)).drop(columns=['day_date']).to_dict('records')}
@app.get('/api/zones')
def geographic(start:str='2025-01-01',end:str='2025-03-31',mode:str='pickup'):
 a,b=bounds(start,end)
 if mode not in ['pickup','dropoff']:raise HTTPException(422,'mode must be pickup or dropoff')
 if mode=='pickup': z=get_zone_data(start,end)[['zone_id','trips']]
 else:z=rows('SELECT do_id zone_id,sum(trips) trips FROM flows WHERE trip_day BETWEEN ? AND ? GROUP BY 1',[a,b])
 lookup={int(x.zone_id):int(x.trips) for x in z.itertuples()}; names={x['id']:x for x in zones()}; g=geometry()
 return {'type':'FeatureCollection','features':[{'type':'Feature','geometry':f['geometry'],'properties':{'id':f['properties']['id'],'name':names.get(f['properties']['id'],{}).get('name','Unknown'),'trips':lookup.get(f['properties']['id'],0)}} for f in g['features']]}
@app.get('/api/temporal')
def temporal(start:str='2025-01-01',end:str='2025-03-31'):
 a,b=bounds(start,end); x=rows('SELECT "hour",sum(trips) trips FROM hourly_zone WHERE CAST("hour" AS DATE) BETWEEN ? AND ? GROUP BY 1 ORDER BY 1',[a,b]); x['h']=x['hour'].dt.hour;x['dow']=x['hour'].dt.dayofweek;x['day']=x['hour'].dt.strftime('%Y-%m-%d');
 return {'hourly':x.groupby('h',as_index=False).trips.mean().round(1).to_dict('records'),'weekday':x.groupby('dow',as_index=False).trips.mean().round(1).to_dict('records'),'heat':x.groupby(['dow','h'],as_index=False).trips.mean().round(1).to_dict('records'),'daily':x.groupby('day',as_index=False).trips.sum().to_dict('records')}
@app.get('/api/flows')
def flows(start:str='2025-01-01',end:str='2025-03-31',origin:int=0):
 a,b=bounds(start,end)
 if not 0<=origin<=263:raise HTTPException(422,'Invalid origin')
 x=rows('SELECT pu,do_id,sum(trips) trips FROM flows WHERE trip_day BETWEEN ? AND ? AND (?=0 OR pu=?) GROUP BY 1,2 ORDER BY trips DESC LIMIT 30',[a,b,origin,origin]); names={z['id']:z['name'] for z in zones()};return {'routes':[{'origin_id':int(r.pu),'destination_id':int(r.do_id),'origin':names.get(int(r.pu),'Unknown'),'destination':names.get(int(r.do_id),'Unknown'),'trips':int(r.trips)} for r in x.itertuples()],'zones':zones()}
@app.get('/api/forecast')
def forecast():
 x=pd.read_parquet(DATA/'forecast.parquet');x['hour']=x.hour.astype(str)
 return {'metrics':json.loads((DATA/'metrics.json').read_text()),'backtests':json.loads((DATA/'backtests.json').read_text()),'predictions':x.to_dict('records')}
@app.get('/api/hotspots')
def hotspots(start:str='2025-01-01',end:str='2025-03-31'):
 a,b=bounds(start,end); x=rows('SELECT zone_id,hour("hour") h,sum(trips) trips FROM hourly_zone WHERE CAST("hour" AS DATE) BETWEEN ? AND ? GROUP BY 1,2',[a,b]); mat=x.pivot(index='zone_id',columns='h',values='trips').fillna(0).reindex(columns=range(24),fill_value=0);mat=mat[mat.sum(axis=1)>=100]
 if len(mat)<4:return {'clusters':[],'profiles':[]}
 p=mat.div(mat.sum(axis=1),axis=0); label=KMeans(n_clusters=4,random_state=42,n_init=10).fit_predict(p); p['cluster']=label; profiles=p.groupby('cluster').mean();return {'clusters':[{'id':int(i),'cluster':int(v)} for i,v in zip(p.index,label)],'profiles':[{'hour':int(h),**{f'cluster_{i}':round(float(profiles.loc[i,h]),4) for i in profiles.index}} for h in range(24)]}
@app.get('/api/data-quality')
def quality():return {'quality':json.loads((DATA/'quality.json').read_text()),'zones':zones()}
@app.get('/api/scenario')
def scenario(start:str='2025-01-01',end:str='2025-03-31',zone:int=0,change:float=Query(0,ge=-50,le=100)):
 z=get_zone_data(start,end)
 if zone and zone not in z.zone_id.values:raise HTTPException(422,'Zone has no pickups in this period')
 base=int(z.loc[z.zone_id==zone,'trips'].sum()) if zone else int(z.trips.sum());return {'baseline':base,'adjusted':base*(1+change/100),'change':change,'label':'Hypothetical proportional adjustment, not a forecast'}
