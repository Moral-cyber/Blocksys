"""Publish checked bilingual papers to the user's single existing paper root."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'
DESKTOP=Path('/Users/wind/Desktop/Codex/paper')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assert DESKTOP.is_dir()
    assert json.loads((OUT/'PAPER_DELIVERY_CHECKS.json').read_text())['status']=='PASS'
    stamp=datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')
    previous=DESKTOP/'01_论文材料/历史论文版本'/stamp
    copied=[]
    for source_name,target_name in (('english_paper','03_英文论文'),('chinese_paper','04_中文论文')):
        source=OUT/source_name;target=DESKTOP/target_name
        assert json.loads((source/'CURRENT_REVISION.json').read_text())['status']=='COMPILED_AND_CHECKED'
        if target.exists():
            backup=previous/target_name;backup.parent.mkdir(parents=True,exist_ok=True)
            target.rename(backup)
        shutil.copytree(source,target)
        for path in source.rglob('*'):
            if path.is_file():
                relative=path.relative_to(source)
                assert sha(path)==sha(target/relative)
                copied.append({'source':str(path),'target':str(target/relative),'sha256':sha(path)})
    for name in ('当前研究状态_请先阅读.md','README.md'):
        target=DESKTOP/'00_使用说明'/name
        if target.exists():
            backup=previous/'00_使用说明'/name;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,backup)
        shutil.copy2(OUT/'当前成果与阅读入口.md',target)
    receipt={'status':'PUBLISHED_AND_HASH_VERIFIED','time':datetime.now().astimezone().isoformat(),
        'destination':str(DESKTOP),'previous_papers_preserved':str(previous),
        'verified_files':len(copied),'files':copied,
        'conversation_archive':'sibling remains unchanged; no new paper root created'}
    (OUT/'PAPER_DESKTOP_RECEIPT.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    (DESKTOP/'00_使用说明/论文同步回执.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'},ensure_ascii=False))

if __name__=='__main__':main()
