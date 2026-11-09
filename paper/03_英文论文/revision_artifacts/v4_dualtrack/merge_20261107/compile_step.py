from pathlib import Path
import subprocess,json,shutil,sys
from pypdf import PdfReader
A=Path(__file__).resolve().parent
tag,description=sys.argv[1:3]
with (A/f"{tag}_compile.log").open("w") as out:
 for cmd in [["/Library/TeX/texbin/pdflatex","-interaction=nonstopmode","-halt-on-error","main.tex"],["/Library/TeX/texbin/bibtex","main"],["/Library/TeX/texbin/pdflatex","-interaction=nonstopmode","-halt-on-error","main.tex"],["/Library/TeX/texbin/pdflatex","-interaction=nonstopmode","-halt-on-error","main.tex"]]:
  r=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
  assert r.returncode==0,(cmd,A/f"{tag}_compile.log")
n=len(PdfReader("main.pdf").pages)
shutil.copy2("main.pdf",A/f"{tag}.pdf");shutil.copy2("main.log",A/f"{tag}.log")
log=json.loads((A/"PAGE_LOG.json").read_text());previous=log["steps"][-1]["pages"]
step={"step":tag,"description":description,"pages_before":previous,"pages":n,"saved_pages":previous-n,"pdf":str(A/f"{tag}.pdf"),"overfull_boxes":Path("main.log").read_text().count("Overfull"),"undefined_references":Path("main.log").read_text().lower().count("undefined")}
log["steps"].append(step);(A/"PAGE_LOG.json").write_text(json.dumps(log,indent=2,ensure_ascii=False)+"\n");print(json.dumps(step,ensure_ascii=False))
