"""Refresh only formatting sources and manifests in the existing research bundle."""
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/method_followup'
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    changed=[]
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        assert sha(source)==sha(target)
        changed.append(target)
    paths=[p for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json')]
    paths+=list((HERE/'template').glob('*'))
    for p in paths:
        relative=p.relative_to(HERE)
        copy(p,OUT/'scripts/blocksys_template'/relative)
        copy(p,OUT/'repro/work/blocksys_template'/relative)
    for p in (ROOT/'work/predictive_manuscript').glob('*.py'):
        copy(p,OUT/'scripts/predictive_manuscript'/p.name)
        copy(p,OUT/'repro/work/predictive_manuscript'/p.name)
    for name in ('sync_method_bundle.py','build_feedback_repro.py'):
        copy(ROOT/'work'/name,OUT/'scripts'/name)
    copy(ROOT/'outputs/当前成果与阅读入口.md',OUT/'00_当前研究状态.md')
    note='''\n\n## 2026-11-04 BlockSys 模板适配完成\n\n依用户确认，2027 年投稿采用给定的 2026 年 svproc 模板。英文14页，中文12页（均含参考文献及附录）；参考按首次引用编号，所有实验数值和核心公式保留。仅进行格式与文字/图表布局适配，未新增实验。脚本及原版class/BST见 scripts/blocksys_template；投稿LaTeX完整包见同级03/04论文目录。\n'''
    for name in ('PROJECT_STATE.md','技能使用与任务接续.md'):
        p=OUT/name
        if 'BlockSys 模板适配完成' not in p.read_text():p.write_text(p.read_text()+note)
        changed.append(p)
    repro=OUT/'repro/repro_source_manifest.json';d=json.loads(repro.read_text());records={r['path']:r for r in d['records']}
    for p in changed:
        if p.is_relative_to(OUT/'repro'):
            relative=str(p.relative_to(OUT/'repro'))
            source=ROOT/relative
            records[relative]={'source':relative,'path':relative,'bytes':p.stat().st_size,'sha256':sha(p)}
    d['records']=sorted(records.values(),key=lambda r:r['path']);d['files']=len(d['records']);d['bytes']=sum(r['bytes'] for r in d['records'])
    d['template_update']='2026 kit selected for2027; formatting source refresh only; no experimental code or data changed'
    repro.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');changed.append(repro)
    manifest=OUT/'bundle_manifest.json';d=json.loads(manifest.read_text());records={r['path']:r for r in d['records']}
    for p in changed:
        relative=str(p.relative_to(OUT));records[relative]={'path':relative,'bytes':p.stat().st_size,'sha256':sha(p)}
    d['records']=sorted(records.values(),key=lambda r:r['path']);d['files']=len(d['records']);d['bytes']=sum(r['bytes'] for r in d['records'])
    d['template_update']={'time':datetime.now().astimezone().isoformat(),'scope':'incremental formatting scripts, unchanged kit, docs, and portable source manifests','files_refreshed':len(changed),'no_experiment_changes':True}
    manifest.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'REFRESHED','changed_files':len(changed),'manifest_files':d['files']}))

if __name__=='__main__':main()
