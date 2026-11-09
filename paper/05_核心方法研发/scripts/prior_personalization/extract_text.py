from pathlib import Path
from html.parser import HTMLParser
class Parse(HTMLParser):
 def __init__(self):super().__init__();self.out=[];self.skip=0;self.math=0
 def handle_starttag(self,tag,attrs):
  attrs=dict(attrs)
  if tag in ['script','style','nav','footer']:self.skip+=1
  if self.skip:return
  if tag=='math':self.math+=1;self.out.append('[MATH '+attrs.get('alttext','')+']');return
  if tag in ['p','div','section','h1','h2','h3','li','tr']:self.out.append('\n')
 def handle_endtag(self,tag):
  if tag in ['script','style','nav','footer']:self.skip=max(0,self.skip-1)
  if self.skip:return
  if tag=='math':self.math=max(0,self.math-1)
  if tag in ['p','div','section','h1','h2','h3','li','tr']:self.out.append('\n')
 def handle_data(self,data):
  if not self.skip and not self.math:self.out.append(data)
for name in ['fedrod','logits_fusion','multi_level']:
 p=Path(__file__).parent/'sources'/(name+'.html');s=Parse();s.feed(p.read_text());txt='\n'.join(x.strip() for x in ''.join(s.out).splitlines() if x.strip());p.with_suffix('.txt').write_text(txt)
 print(name,len(txt))
