import json, unittest
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app
P=Path(__file__).resolve().parents[1]/'data'/'processed'
class ApiTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.c=TestClient(app);cls.q=json.loads((P/'quality.json').read_text());cls.m=json.loads((P/'metrics.json').read_text())
 def test_reconciliation(self):
  x=self.c.get('/api/overview').json();self.assertEqual(x['trips'],self.q['cleaned']); self.assertEqual(sum(z['trips'] for z in x['daily']),self.q['cleaned'])
 def test_geography_and_dropoff(self):
  for mode in ('pickup','dropoff'):
   r=self.c.get('/api/zones',params={'mode':mode});self.assertEqual(r.status_code,200);self.assertGreater(len(r.json()['features']),250)
 def test_forecast_integrity(self):
  f=self.c.get('/api/forecast').json();self.assertEqual(len(f['predictions']),self.m['test_n']);self.assertEqual(f['metrics']['model']['MAE'],self.m['model']['MAE']);self.assertEqual(len(f['backtests']),4)
 def test_hypothetical_math(self):
  x=self.c.get('/api/scenario',params={'change':10}).json();self.assertAlmostEqual(x['adjusted'],x['baseline']*1.1)
 def test_invalid_inputs(self):
  self.assertEqual(self.c.get('/api/overview',params={'start':'2024-01-01'}).status_code,422);self.assertEqual(self.c.get('/api/zones',params={'mode':'other'}).status_code,422);self.assertEqual(self.c.get('/api/scenario',params={'change':200}).status_code,422)
 def test_filtered_flow(self):
  x=self.c.get('/api/flows',params={'origin':161}).json();self.assertTrue(x['routes']);self.assertTrue(all(r['origin_id']==161 for r in x['routes']))
 def test_profile_clusters(self):
  x=self.c.get('/api/hotspots').json();self.assertGreater(len(x['clusters']),3);self.assertEqual(len(x['profiles']),24)
if __name__=='__main__':unittest.main()
