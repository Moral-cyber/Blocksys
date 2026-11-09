from pathlib import Path
import urllib.request,urllib.parse,json,time
r=Path(__file__).resolve().parent
qs=['direction aware gradient sparsification','gradient alignment sparsification federated learning','global gradient guided sparsification','consensus gradient sparsification','energy constrained gradient compression','Gradient-Congruity Guided Federated Sparse Training']
out=[]
for i,q in enumerate(qs):
 url='https://api.crossref.org/works?'+urllib.parse.urlencode({'query.bibliographic':q,'rows':8})
 rec={'query':q,'url':url,'date':'2026-11-03'}
 try:
  raw=urllib.request.urlopen(url,timeout=30).read();(r/f'alignment_crossref_{i:02d}.json').write_bytes(raw);msg=json.loads(raw)['message'];rec['entries']=[{k:e.get(k) for k in ['title','DOI','author','published','container-title','URL','abstract']} for e in msg['items']];print(json.dumps({'query':q,'titles':[(e['title'],e['DOI']) for e in rec['entries']]},ensure_ascii=False),flush=True)
 except Exception as e:rec['error']=str(e)
 out.append(rec);(r/'alignment_crossref_search_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));time.sleep(.3)
