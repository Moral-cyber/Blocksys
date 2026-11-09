"""Render actual final PDF pages and inventory page/font properties."""
from pathlib import Path
import hashlib,json
import pypdfium2 as pdfium
from pypdf import PdfReader
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'
DEST=ROOT/'work/blocksys_template/page_renders'
DEST.mkdir(parents=True,exist_ok=True)
for lang in ('english','chinese'):
    path=OUT/f'{lang}_paper/main.pdf';doc=pdfium.PdfDocument(path);reader=PdfReader(path)
    pages=[];thumbs=[]
    folder=DEST/lang;folder.mkdir(exist_ok=True)
    fonts=[]
    for i,page in enumerate(reader.pages):
        pic=doc[i].render(scale=1.4).to_pil().convert('RGB');pic.save(folder/f'{i+1:02}.png')
        thumb=pic.copy();thumb.thumbnail((306,396));thumbs.append(thumb)
        pages.append({'page':i+1,'size_pt':[float(page.mediabox.width),float(page.mediabox.height)],'text_chars':len(page.extract_text()),'text_start':page.extract_text()[:100]})
        for name,ref in page['/Resources'].get('/Font',{}).items():
            f=ref.get_object();d=f.get('/DescendantFonts',[f])[0].get_object();fd=d.get('/FontDescriptor',{});fd=fd.get_object() if hasattr(fd,'get_object') else fd
            fonts.append({'base':str(f.get('/BaseFont')),'subtype':str(f.get('/Subtype')),'embedded':any(k in fd for k in ('/FontFile','/FontFile2','/FontFile3'))})
    sheet=Image.new('RGB',(4*326,((len(thumbs)+3)//4)*422),'#dedede');draw=ImageDraw.Draw(sheet)
    for i,pic in enumerate(thumbs):
        x=(i%4)*326+10;y=(i//4)*422+20;sheet.paste(pic,(x,y));draw.text((x,y-16),f'{lang} page {i+1}',fill='black')
    sheet.save(DEST/f'{lang}_contact.png')
    unique={f['base']:f for f in fonts}
    report={'language':lang,'pdf_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pages':pages,'fonts':list(unique.values()),'contact_sheet':str(DEST/f'{lang}_contact.png')}
    (DEST/f'{lang}_INVENTORY.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'language':lang,'pages':len(doc),'all_top_fonts_embedded':all(f['embedded'] for f in fonts),'contact_sheet':report['contact_sheet']}))
