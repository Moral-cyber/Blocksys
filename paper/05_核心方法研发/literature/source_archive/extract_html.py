from pathlib import Path
from html.parser import HTMLParser
class P(HTMLParser):
 def __init__(self):super().__init__();self.s=[];self.skip=0
 def handle_starttag(self,t,a):
  if t in ['script','style']:self.skip+=1
  if t in ['p','h1','h2','h3','h4','section','table','li']:self.s.append('\n')
 def handle_endtag(self,t):
  if t in ['script','style']:self.skip=max(0,self.skip-1)
 def handle_data(self,d):
  if not self.skip:self.s.append(d)
for p in Path(__file__).resolve().parent.glob('*.html'):
 q=P();q.feed(p.read_text());p.with_suffix('.txt').write_text(''.join(q.s))
