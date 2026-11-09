from pathlib import Path
import urllib.request,urllib.parse,xml.etree.ElementTree as ET,json,time
root=Path(__file__).resolve().parent
qs=['ti:"Data-Aware Gradient Compression"','ti:"MultiGS"','ti:"GRASP" AND abs:"sparsification"','ti:"Communication-Adaptive-Gradient"']
ns={'a':'http://www.w3.org/2005/Atom','x':'http://arxiv.org/schemas/atom','o':'http://a9.com/-/spec/opensearch/1.1/'};out=[]
for i,q in enumerate(qs):
 url='https://export.arxiv.org/api/query?'+urllib.parse.urlencode({'search_query':q,'max_results':10,'sortBy':'relevance'});r={'query':q,'url':url,'date':'2026-11-03','entries':[]}
 try:
  raw=urllib.request.urlopen(url,timeout=30).read();(root/f'exact_neighbor_{i:02d}.xml').write_bytes(raw);tree=ET.fromstring(raw);r['total']=tree.findtext('o:totalResults',namespaces=ns)
  for e in tree.findall('a:entry',ns):
   r['entries'].append({k:' '.join(e.findtext(v,default='',namespaces=ns).split()) for k,v in [('title','a:title'),('id','a:id'),('published','a:published'),('summary','a:summary'),('doi','x:doi'),('journal_ref','x:journal_ref')]}|{'authors':[a.findtext('a:name',namespaces=ns) for a in e.findall('a:author',ns)]})
  print(json.dumps(r,ensure_ascii=False),flush=True)
 except Exception as e:r['error']=str(e);print(json.dumps(r),flush=True)
 out.append(r);(root/'exact_neighbor_search_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));time.sleep(3.2)
