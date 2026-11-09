from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image,ImageDraw
r=Path('/private/tmp/pr_lfl_route_a/final_visual');r.mkdir(exist_ok=True)
pdf=pdfium.PdfDocument('/private/tmp/pr_lfl_route_a/build/main.pdf')
ims=[]
for i,p in enumerate(pdf):
 im=p.render(scale=130/72).to_pil().convert('RGB');im.save(r/f'page_{i+1:02}.png');ims.append(im)
for a in range(0,len(ims),4):
 thumbs=[]
 for j,im in enumerate(ims[a:a+4]):
  im=im.copy();im.thumbnail((650,900));thumbs.append(im)
 sheet=Image.new('RGB',(1340,1880),'#c9c9c9');d=ImageDraw.Draw(sheet)
 for j,im in enumerate(thumbs):
  x=10+(j%2)*670;y=25+(j//2)*935;sheet.paste(im,(x,y));d.text((x+5,y-18),f'Page {a+j+1}',fill='black')
 sheet.save(r/f'sheet_{a+1:02}_{min(a+4,len(ims)):02}.png')
print(len(pdf),r)
