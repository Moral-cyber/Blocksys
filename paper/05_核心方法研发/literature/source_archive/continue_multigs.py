from pathlib import Path
import urllib.request,concurrent.futures,json
root=Path(__file__).resolve().parent
urls=[
 ('multigs_ieee','https://ieeexplore.ieee.org/document/11267482/'),
 ('multigs_ieee_metadata','https://ieeexplore.ieee.org/rest/document/11267482/metadata'),
 ('multigs_openalex','https://api.openalex.org/works/https://doi.org/10.1109/TCSVT.2025.3636885'),
 ('multigs_s2_retry','https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TCSVT.2025.3636885?fields=title,authors,year,abstract,url,openAccessPdf,externalIds'),
 ('multigs_bing','https://www.bing.com/search?q=%22MultiGS%22+%22gradient%22'),
 ('multigs_ddg','https://html.duckduckgo.com/html/?q=%22MultiGS%22+%22gradient%22')]
def f(x):
 name,url=x;rec={'name':name,'url':url,'date':'2026-11-03'}
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (Academic literature verification)'})
  resp=urllib.request.urlopen(req,timeout=30);d=resp.read();ct=resp.headers.get('Content-Type','');ext='.json' if 'json' in ct else '.html';(root/(name+ext)).write_bytes(d);rec|={'status':resp.status,'bytes':len(d),'file':name+ext,'content_type':ct};print(json.dumps(rec),flush=True)
 except Exception as e:rec['error']=str(e);print(json.dumps(rec),flush=True)
 return rec
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:r=list(ex.map(f,urls))
(root/'multigs_followup_retrievals.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
