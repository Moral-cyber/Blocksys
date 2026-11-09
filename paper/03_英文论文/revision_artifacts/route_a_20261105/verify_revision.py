"""Read-only checks for the route-A revision; no training or result rewriting."""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, re
from pypdf import PdfReader

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args()
    root = args.paper
    lock = json.loads((Path(__file__).parent/'baseline_lock.json').read_text())
    backup = Path(lock['backup'])/'03_英文论文'
    protected = [k for k in lock['sha256'] if k.startswith(('evidence/','checks/'))
                 or k.endswith('.provenance.json') or k in
                 ['svproc.cls','spmpsci_unsrt.bst','figures/FIGURE_PROVENANCE.json']]
    protected_bad = [k for k in protected if sha(root/k) != lock['sha256'][k]]
    preamble_same = (root/'main.tex').read_bytes().split(b'\\begin{document}')[0] == (backup/'main.tex').read_bytes().split(b'\\begin{document}')[0]
    bib_prefix = (root/'references.bib').read_bytes().startswith((backup/'references.bib').read_bytes())
    preserved = {}
    for name in ['main','training','quality']:
        old = (backup/f'tables/{name}.tex').read_text()
        new = (root/f'tables/{name}.tex').read_text()
        if name == 'main': new += (root/'tables/main_full.tex').read_text()
        cells = lambda s: Counter(x for x in re.findall(r'\$([^$]+)\$',s) if re.search(r'\d',x))
        preserved[name] = not bool(cells(old)-cells(new))
    preserved['seeds_byte_identical'] = sha(root/'tables/seeds.tex') == lock['sha256']['tables/seeds.tex']
    sections = {p.name:p.read_text() for p in (root/'sections').glob('*.tex')}
    all_tex = '\n'.join(sections.values())+'\n'+'\n'.join(p.read_text() for p in (root/'tables').glob('*.tex'))
    cited = set(k for s in re.findall(r'\\cite\{([^}]+)\}',all_tex) for k in s.split(','))
    bib_keys = set(re.findall(r'@\w+\{([^,]+),',(root/'references.bib').read_text()))
    abstract = sections['abstract.tex'].split(r'\begin{abstract}')[1].split(r'\keywords')[0]
    abstract_words = len(abstract.split())
    pattern = r'does not|do not|rather than|only|remain|without|not a |scope|separate|trusted|exploratory|disclosed'
    literal_hits = [{'file':n,'line':i,'text':line} for n,s in sections.items() for i,line in enumerate(s.splitlines(),1) if re.search(pattern,line)]
    ending_hits=[]
    for n,s in sections.items():
        for para in re.split(r'\n\s*\n',s):
            if not para.strip() or para.lstrip().startswith(('\\begin','\\include','\\caption','\\input','\\section','\\subsection')): continue
            sentence = re.split(r'(?<=[.!?])\s+',para.strip())[-1]
            if re.search(pattern,sentence): ending_hits.append({'file':n,'last_sentence':sentence})
    main_text = (root/'tables/main.tex').read_text()
    main_rows = [l for l in main_text.splitlines() if '&' in l and '$' in l]
    main_tables = 0
    for n,s in sections.items():
        if n=='appendix.tex':continue
        main_tables += s.count(r'\begin{table}')
        for rel in re.findall(r'\\input\{([^}]+)\}',s):
            main_tables += (root/(rel+'.tex')).read_text().count(r'\begin{table}')
    pdf = PdfReader(args.build/'main.pdf')
    log = (args.build/'main.log').read_text()
    problems = [l for l in log.splitlines() if re.search(r'Overfull|undefined|LaTeX Warning:|^!|Missing character',l)]
    banned = re.findall(r'—|innovative|superior|surpass|notably|pave the way|underscore',all_tex)
    qa = re.findall(r'128 assertions|\b291\b|3\.67e-16|manually mined',all_tex)
    checks = {'protected_files_unchanged':not protected_bad,'preamble_unchanged':preamble_same,
        'old_bibliography_byte_prefix':bib_prefix,'original_table_numbers_preserved':all(preserved.values()),
        'abstract_at_most_150':abstract_words<=150,'at_least_30_cited':len(cited)>=30,
        'all_citations_defined':not bool(cited-bib_keys),'two_main_tables':main_tables==2,
        'six_main_rows':len(main_rows)==6,'at_most_eight_paragraph_endings':len(ending_hits)<=8,
        'at_most_eight_literal_scan_lines':len(literal_hits)<=8,'no_banned_language':not banned,
        'no_internal_QA_numbers':not qa,'citation_spaces_correct':not bool(re.search(r' \\cite\{',all_tex)),
        'at_most_12_pages':len(pdf.pages)<=12,'clean_final_log':not problems}
    result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
        'counts':{'pages':len(pdf.pages),'abstract_whitespace_words':abstract_words,'cited':len(cited),
        'bibliography_entries':len(bib_keys),'main_tables':main_tables,'main_rows':len(main_rows),
        'literal_scan_lines':len(literal_hits),'paragraph_ending_hits':len(ending_hits)},
        'protected_file_count':len(protected),'protected_mismatches':protected_bad,
        'original_table_checks':preserved,'undefined_citation_keys':sorted(cited-bib_keys),
        'final_log_problems':problems,'literal_scan':literal_hits,'paragraph_ending_scan':ending_hits,
        'pdf_sha256':sha(args.build/'main.pdf')}
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
