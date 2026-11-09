from pathlib import Path
import urllib.request,urllib.parse,json,time
root=Path(__file__).resolve().parent
qs=['gradient sparsification global direction','gradient sparsification energy norm constraint','history guided gradient sparsification','direction aware gradient compression']
out=[]
for i,q in enumerate(qs):
 url='https://api.openalex.org/works?'+urllib.parse.urlencode({'search':q,'per_page':15,'select':'id,title,doi,publication_year,abstract_inverted_index,primary_location,open_access'})
 r={'query':q,'url':url,'date':'2026-11-03'}
 try:
  raw=urllib.request.urlopen(url,timeout=30).read();(root/f'guarded_openalex_{i:02d}.json').write_bytes(raw);m=json.loads(raw);r['total']=m.get('meta',{}).get('count');r['entries']=m.get('results',[]);print(json.dumps({'query':q,'total':r['total'],'titles':[(e['title'],e['doi']) for e in r['entries']]},ensure_ascii=False),flush=True)
 except Exception as e:r['error']=str(e);print(json.dumps(r),flush=True)
 out.append(r);(root/'guarded_openalex_search_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));time.sleep(.5)
