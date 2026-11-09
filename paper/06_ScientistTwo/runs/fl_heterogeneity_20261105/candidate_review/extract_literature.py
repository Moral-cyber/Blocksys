"""Extract already-retrieved neighbor sources; no network or experiment use."""
from pathlib import Path
import hashlib
from html.parser import HTMLParser
import html
import json
import re

PAPER = Path('/Users/wind/Desktop/Codex/paper')
HERE = Path(__file__).resolve().parent
LITERATURE = Path('/private/tmp/pr_lfl_route_a/03_英文论文/revision_artifacts/route_a_20261105/literature/near_neighbors')

def identity(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

source = Path('/private/tmp/pr_lfl_route_a/literature/saerens2002.json')
metadata = json.loads(source.read_text())['message']
saerens = {'id': 'saerens2002', 'source': identity(source), 'doi': metadata['DOI'],
           'url': metadata['resource']['primary']['URL'],
           'verified_reading_scope': 'publisher/Crossref abstract and metadata, no full-text equation claim',
           'abstract': re.sub('<[^>]+>', '', metadata['abstract']),
           'implication': 'Adjusting classifier outputs for changed class priors without model refit is prior art; the ratio formula in review is derived directly from Bayes.'}

source = PAPER / '05_核心方法研发/literature/prior_personalization/sources/menon_la.txt'
lines = source.read_text().splitlines()
start = next(i for i, line in enumerate(lines) if line.startswith('3 Logit adjustment for long-tail learning'))
end = next(i for i, line in enumerate(lines[start + 1:], start + 1) if line.startswith('4 Post-hoc logit adjustment'))
menon = {'id': 'menon2021', 'source': identity(source), 'url': 'https://arxiv.org/abs/2007.07314v2',
         'verified_reading_scope': 'full-text section3 Eq8 and transition to section4',
         'line_start': start + 1, 'line_end': end + 1, 'excerpt': '\n'.join(lines[start:end + 1]),
         'implication': 'Subtracting log source priors is classic post-hoc logit adjustment; balanced-error goal does not establish the local-q estimator or NLL claim.'}

class Capture(HTMLParser):
    def __init__(self, equation):
        super().__init__(); self.equation = equation; self.depth = 0; self.active = False; self.text = []; self.annotations = []; self.annotation = None
    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if not self.active and attributes.get('id') == self.equation:
            self.active = True; self.depth = 1
        elif self.active:
            self.depth += 1
        if self.active and tag == 'annotation' and attributes.get('encoding') == 'application/x-tex':
            self.annotation = []
    def handle_data(self, value):
        if self.active:
            self.text.append(value)
            if self.annotation is not None: self.annotation.append(value)
    def handle_endtag(self, tag):
        if self.active:
            if tag == 'annotation' and self.annotation is not None:
                self.annotations.append(''.join(self.annotation)); self.annotation = None
            self.depth -= 1
            if self.depth == 0: self.active = False

source = LITERATURE / 'prior_full.html'
payload = source.read_text()
equations = {}
for key in ('S4.E8', 'S4.E11', 'S5.E14'):
    if f'id="{key}"' not in payload:
        candidates = re.findall(r'id="([^"]+\.E' + key.split('E')[-1] + ')"', payload)
        if len(candidates) != 1:
            raise ValueError('one exact equation required: ' + key)
        key = candidates[0]
    captured = Capture(key); captured.feed(payload)
    if not captured.text or not captured.annotations:
        raise ValueError('equation capture empty: ' + key)
    equations[key] = {'text': ' '.join(' '.join(captured.text).split()),
                      'tex_annotations': captured.annotations}
prior = {'id': 'shi2023prior', 'source': identity(source), 'url': 'https://arxiv.org/html/2310.09183v2',
         'verified_reading_scope': 'full-text exact equations8,11,14; model-space personalized prior framework',
         'equations': equations,
         'implication': 'Personalized regularization centers already appear in model training; a local output-marginal denominator is not provided by this source.'}

out = {'status': 'EXTRACTED_FROM_EXISTING_PRIMARY_SOURCES', 'extractor': identity(Path(__file__)),
       'sources': [saerens, menon, prior], 'new_network_search': False,
       'scope': 'bounded neighbor comparison, not a novelty search or claim of originality'}
(HERE / 'LITERATURE_EXTRACT.json').write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'status': out['status'], 'sources': [v['id'] for v in out['sources']],
                  'prior_equation_ids': list(equations)}))
