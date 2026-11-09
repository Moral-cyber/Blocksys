from pathlib import Path
import urllib.request, urllib.parse, json,time
root=Path(__file__).resolve().parent
queries=['FedGraVAC','FedSparsify','FedMask','FedCSPACK','Fed CS PACK federated learning','heterogeneity aware sparsification federated learning','global direction gradient sparsification federated learning','gradient cancellation sparsification federated learning','FedDST']
records=[]
for i,q in enumerate(queries):
    url='https://api.crossref.org/works?'+urllib.parse.urlencode({'query.bibliographic':q,'rows':5})
    rec={'query':q,'url':url,'date':'2026-11-03'}
    try:
        raw=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'AcademicEvidenceCheck/1.0'}),timeout=30).read()
        (root/f'crossref_{i:02d}.json').write_bytes(raw)
        msg=json.loads(raw)['message'];rec['total']=msg['total-results'];rec['entries']=[{k:e.get(k) for k in ['title','DOI','author','published','container-title','URL','abstract']} for e in msg['items']]
        print(json.dumps({'query':q,'titles':[(e['title'],e['DOI']) for e in rec['entries']]},ensure_ascii=False),flush=True)
    except Exception as e:rec['error']=str(e);print(json.dumps(rec),flush=True)
    records.append(rec);(root/'crossref_search_results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    time.sleep(.3)
