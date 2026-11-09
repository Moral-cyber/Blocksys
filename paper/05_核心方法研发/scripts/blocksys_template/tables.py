"""Reflow wide tables from preserved row cells, without rounding metrics."""
import re

def apply(folder):
    for name in ('main','training'):
        path=folder/f'tables/{name}.tex';text=path.read_text()
        head=text[:text.index('\\begin{tabular}')]
        data=text.split('\\midrule',1)[1].split('\\bottomrule',1)[0]
        rows=[line.strip().removesuffix('\\\\').split(' & ') for line in data.splitlines() if ' & ' in line]
        # Remove presentation-only stacks before using a three-column layout.
        for row in rows:
            row[0]=row[0].replace(r'\shortstack[l]{FedLC-inspired\\+ uniform3}','FedLC-inspired + uniform3')
            row[0]=row[0].replace(r'\shortstack[l]{FedLC-inspired\\+ dense}','FedLC-inspired + dense')
            assert len(row)==5
        parts=[]
        for index,task in enumerate(('CIFAR-10','FashionMNIST')):
            parts.append('\\begin{tabular}{lrr}\n\\toprule\n'+
                '\\multicolumn{3}{l}{'+task+'}\\\\\n'+
                ('Predictor' if name=='main' else 'Training')+r' & Acc. (\%) & NLL\\'+'\n\\midrule\n')
            parts.extend(' & '.join([row[0],*row[1+2*index:3+2*index]])+'\\\\\n' for row in rows)
            parts.append('\\bottomrule\n\\end{tabular}\n')
            if index==0:parts.append('\\par\\medskip\n')
        path.write_text(head+''.join(parts)+'\\end{table}\n')
    # Separate task labels into full-width group rows, increasing numeric
    # column space while retaining all original values and seed identifiers.
    for name,columns,heading in (
        ('quality','lrrr',r'Predictor & Brier & ECE$_{\rm macro}$ & ECE$_{\rm pooled}$\\'),
        ('seeds','lrrrr',r'Seed & Acc. (\%) & NLL & $\Delta$Acc. (pp) & $-\Delta$NLL\\')):
        path=folder/f'tables/{name}.tex';text=path.read_text()
        head=text[:text.index('\\begin{tabular}')]
        data=text.split('\\midrule',1)[1].split('\\bottomrule',1)[0]
        rows=[line.strip().removesuffix('\\\\').split(' & ') for line in data.splitlines() if ' & ' in line]
        content=['\\begin{tabular}{'+columns+'}\n\\toprule\n'+heading+'\n']
        previous=None
        for row in rows:
            if row[0]!=previous:
                previous=row[0]
                task='FashionMNIST' if previous=='Fashion' else previous
                content.append('\\midrule\n\\multicolumn{'+str(len(row)-1)+'}{l}{'+task+'}\\\\\n')
            content.append(' & '.join(row[1:])+'\\\\\n')
        content.append('\\bottomrule\n\\end{tabular}\n\\end{table}\n')
        path.write_text(head+''.join(content))
