from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,json
r=Path('/private/tmp/scientisttwo_sources/near');u='https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_FedNPC_Stochastic_Noise-driven_Post-hoc_Classifier_Calibration_Method_for_Federated_Long-tailed_CVPRF_2026_paper.pdf';size=1643740;chunk=140000
parts=[(i,min(i+chunk-1,size-1)) for i in range(0,size,chunk)]
def get(ab):
 a,b=ab;p=r/f'fednpc_chunk_{a}.part';h=r/f'fednpc_chunk_{a}.headers';ret=subprocess.run(['curl','-sS','-L','--fail','--max-time','55','--range',f'{a}-{b}','-D',str(h),'-o',str(p),u],capture_output=True);return {'start':a,'end':b,'exit':ret.returncode,'size':p.stat().st_size if p.exists() else 0,'stderr':ret.stderr.decode()}
res=list(ThreadPoolExecutor(max_workers=12).map(get,parts));(r/'FEDNPC_FETCH.json').write_text(json.dumps(res,indent=2));print(json.dumps(res))
if all(x['exit']==0 and x['size']==x['end']-x['start']+1 for x in res):
 (r/'fednpc_cvf_full.pdf').write_bytes(b''.join((r/f'fednpc_chunk_{a}.part').read_bytes() for a,b in parts));print('ASSEMBLED')
