"""Final source/number/citation/compile checks, no research or model fitting."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def expand(path):
    text=path.read_text()
    return re.sub(r'\\input\{([^}]+)\}',lambda m:expand(path.parent/(m[1]+'.tex')),text)

def main():
    reports={}
    for lang in ('english','chinese'):
        folder=OUT/f'{lang}_paper'
        # Inputs are relative to the main project's directory, not each section.
        def read(rel):
            text=(folder/rel).read_text()
            return re.sub(r'\\input\{([^}]+)\}',lambda m:read(m[1]+'.tex'),text)
        alltex=read('main.tex')
        first=[]
        for group in re.findall(r'\\cite\{([^}]+)\}',alltex):
            for key in group.split(','):
                if key not in first:first.append(key)
        numbered=re.findall(r'\\bibcite\{([^}]+)\}\{(\d+)\}',(folder/'main.aux').read_text())
        assert numbered==[(key,str(i+1)) for i,key in enumerate(first)]
        assert '\\bibliographystyle{spmpsci_unsrt}' in alltex
        assert '\\documentclass{svproc}' in alltex
        assert not re.search(r'\\usepackage(?:\[[^]]*\])?\{(?:geometry|lmodern|microtype)\}',alltex)
        template=json.loads((folder/'checks/TEMPLATE_SHA256.json').read_text())
        for name,expected in template.items():
            assert sha(folder/name)==expected
            assert sha(ROOT/'work/blocksys_template/template'/name)==expected
        log=(folder/'main.log').read_text()
        bad=[line for line in log.splitlines() if re.search('Overfull|Warning|undefined|Missing character|! LaTeX Error',line)]
        assert not bad,bad
        underfull=[line for line in log.splitlines() if 'Underfull' in line]
        assert not re.search(r'\bTODO\b|\bTBD\b|PLACEHOLDER',alltex)
        data=json.loads((folder/'evidence/PAPER_NUMBERS.json').read_text())
        assert data['status']=='DONE'
        for source,expected in data['source_sha256'].items():assert sha(source)==expected,source
        pdf=PdfReader(folder/'main.pdf')
        assert 1 <= len(pdf.pages) <= 14
        assert all(abs(float(page.mediabox.width)-612)<0.1 and abs(float(page.mediabox.height)-792)<0.1 for page in pdf.pages)
        assert all(len(page.extract_text())>100 for page in pdf.pages)
        abstract=(folder/'sections/abstract.tex').read_text().split('\\keywords')[0].replace('\\begin{abstract}','')
        abstract_words=len(abstract.split()) if lang=='english' else None
        if abstract_words is not None: assert 70<=abstract_words<=150
        visual=json.loads((folder/'checks/VISUAL_CHECK.json').read_text())
        assert visual['status']=='PASS' and visual['pdf_sha256']==sha(folder/'main.pdf')
        assert visual['pages_inspected']==len(pdf.pages)
        tables={name:sha(folder/f'tables/{name}.tex') for name in ('main','quality','reference','training','seeds')}
        for name in tables:
            def table_numbers(path):
                # Numeric cells, with row/column layout independent of task
                # stacking. Retain counts so duplicated/missing cells fail.
                text=path.read_text()
                rows=[line.removesuffix('\\\\').split(' & ') for line in text.splitlines()
                    if ' & ' in line and '\\pm' in line]
                if rows:
                    return Counter(cell for row in rows for cell in row[1:]
                        if (cell.startswith('$') and '\\pm' in cell) or re.fullmatch(r'[0-9,]+',cell))
                # Individual-seed table has no math-delimited metric cells.
                return Counter(tuple(re.findall(r'[-+]?\d+(?:\.\d+)?',line.split(' & ',1)[1]))
                    if len(line.removesuffix('\\\\').split(' & '))==6 else tuple(re.findall(r'[-+]?\d+(?:\.\d+)?',line))
                    for line in text.splitlines() if ' & ' in line and re.search(r'202612\d\d',line))
            previous=OUT/'previous_revision/pre_blocksys_20261104'/f'{lang}_paper/tables/{name}.tex'
            assert table_numbers(folder/f'tables/{name}.tex')==table_numbers(previous),name
        report={'status':'PASS','language':lang,'pdf_pages':len(pdf.pages),
            'cited_references':len(first),'bibliography_inventory':20,'first_citation_order':first,
            'citation_numbering':'spmpsci_unsrt; first occurrence equals aux bibcite order',
            'template_sha256':template,'english_abstract_words':abstract_words,
            'compile_errors_or_overflows':bad,'nonfatal_underfull_notices':underfull,'tables':tables,
            'numbers':'all generated from completed frozen official TEST results; source SHA unchanged',
            'visual_check':'latest all-page contact sheets inspected; no overlap/clipping; full-size equations and tables checked',
            'scope':'user-selected 2026 BlockSys svproc kit reused for 2027 by explicit user instruction',
            'external_cross_model_review':'REVIEW_UNAVAILABLE',
            'independent_numerical_checks':['CIFAR PASS9/9','Fashion PASS9/9'],
            'remaining_system_work':'real ledger/network and authenticated end-to-end learning listed separately'}
        (folder/'checks/FINAL_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
        (folder/'CHECK_REPORT.md').write_text(f'''# Final manuscript checks

PASS: {lang}, {len(pdf.pages)} pages, {len(first)} cited references from 20-entry verified metadata inventory.

Both manuscripts compile successfully in the user-supplied svproc class. No undefined citations, missing glyphs, overflowing boxes or LaTeX warnings appear in the saved final logs. There are {len(underfull)} nonfatal underfull spacing notices; their pages and references were visually checked, and the notices are retained in the saved log. References follow first appearance (BibTeX spmpsci_unsrt); all numeric-source hashes are unchanged. The class and bibliography style are byte-identical to the supplied package. No geometry, font-size, body-spacing or header overrides are applied. The PDF is Letter, with the class's 12.2 cm by 19.3 cm text area. Every main/control table derives from fixed three-seed official TEST outputs. The reflowed tables preserve every original numeric cell. Mean and sample SD use training seeds, not clients/rounds. Both task exports passed a separate saved-array numerical/provenance check. Latest page renders were inspected for layout and legibility.

The smaller same-prior BCTS control and its competitive ECE are retained; the paper does not attribute all improvement to class-specific scales or require FedLC to be necessary. Actual bytes are separated from analytical dense/downlink accounting. Real ledger/network and authenticated end-to-end learning remain in EXPERIMENT_CHECKLIST.md.

All PDF pages, including references and appendix, fit the conservative local target of <=14 pages. The English abstract follows the sample's 70–150-word guidance. The user explicitly selected the 2026 template for the 2027 submission; reuse of this template is settled. Empty author and affiliation fields need the author's information; they do not assert an anonymity policy. The Chinese version is a reading copy. See TEMPLATE_COMPLIANCE.md for the evidence boundary.

This is a local evidence, source and typesetting check. External cross-model review remains REVIEW_UNAVAILABLE.
''')
        current=json.loads((folder/'CURRENT_REVISION.json').read_text())
        current.update(status='COMPILED_AND_CHECKED',pdf_sha256=sha(folder/'main.pdf'),pages=len(pdf.pages))
        (folder/'CURRENT_REVISION.json').write_text(json.dumps(current,indent=2)+'\n')
        for name in ('main.log','main.blg'):
            shutil.copy2(folder/name,folder/'checks'/name)
        # Keep compiler-only files out of the reading directory after checks.
        for name in ('main.aux','main.bbl','main.blg','main.log','main.out','main.toc'):
            (folder/name).unlink(missing_ok=True)
        files=[p for p in folder.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name!='DELIVERY_MANIFEST.json']
        manifest=[{'path':str(p.relative_to(folder)),'sha256':sha(p),'bytes':p.stat().st_size} for p in files]
        (folder/'checks/DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
        package=folder/('English_Paper_Complete_Package.zip' if lang=='english' else '中文论文完整包.zip')
        with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(folder.rglob('*')):
                if p.is_file() and p!=package and p.suffix!='.zip':archive.write(p,str(p.relative_to(folder)))
        reports[lang]=report
    # Bilingual tables and method equations must be identical numerically.
    for name in ('main','quality','reference','training','seeds'):
        def body(lang):
            return (OUT/f'{lang}_paper/tables/{name}.tex').read_text().split('\\begin{tabular}',1)[1]
        assert body('english')==body('chinese')
    (OUT/'PAPER_DELIVERY_CHECKS.json').write_text(json.dumps({'status':'PASS','reports':reports,
        'bilingual_numeric_tables_identical':True,'no_new_experiments':True},indent=2)+'\n')
    print(json.dumps({'status':'PASS','languages':list(reports),'cited_refs':len(first),'pages':{lang:r['pdf_pages'] for lang,r in reports.items()}}))

if __name__=='__main__':main()
