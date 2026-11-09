"""Compile only existing manuscript sources; never run learning experiments."""
from pathlib import Path
import argparse,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--language',choices=('english','chinese','both'),default='both');args=parser.parse_args()
    langs=('english','chinese') if args.language=='both' else (args.language,)
    report=[]
    for lang in langs:
        folder=ROOT/'outputs'/f'{lang}_paper'
        if lang=='english':
            engine=shutil.which('pdflatex') or '/Library/TeX/texbin/pdflatex'
            bib=shutil.which('bibtex') or '/Library/TeX/texbin/bibtex'
            commands=[[engine,'-interaction=nonstopmode','-halt-on-error','main.tex'],[bib,'main'],
                [engine,'-interaction=nonstopmode','-halt-on-error','main.tex'],[engine,'-interaction=nonstopmode','-halt-on-error','main.tex']]
        else:
            tectonic=shutil.which('tectonic') or '/Users/wind/.local/bin/tectonic'
            commands=[[tectonic,'--keep-logs','--keep-intermediates','main.tex']]
        records=[]
        for command in commands:
            result=subprocess.run(command,cwd=folder,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            records.append({'command':command,'returncode':result.returncode,'output':result.stdout})
            if result.returncode:
                (HERE/f'{lang}_compile.json').write_text(json.dumps(records,indent=2)+'\n')
                raise RuntimeError(result.stdout[-6000:])
        (HERE/f'{lang}_compile.json').write_text(json.dumps(records,indent=2)+'\n')
        report.append({'language':lang,'status':'COMPILED','pdf':str(folder/'main.pdf')})
    print(json.dumps(report))

if __name__=='__main__':main()
