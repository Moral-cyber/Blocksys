from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42}):
 fig,ax=plt.subplots(figsize=(7,3.7),layout='constrained');ax.set_xlim(-.6,3.5);ax.set_ylim(-.4,8.5);ax.axis('off')
 xs=[0,1.55,3.1]
 for x,title in zip(xs,['Client','Coordinator + durable state','EVM registry']):
  ax.text(x,8.2,title,ha='center',va='center',weight='bold',fontsize=9);ax.plot([x,x],[0,7.8],color='.75',lw=1,ls='--')
 def arrow(x1,x2,y,label):
  ax.annotate('',xy=(x2,y),xytext=(x1,y),arrowprops={'arrowstyle':'->','color':'#0072B2','lw':1.3});ax.text((x1+x2)/2,y+.14,label,ha='center',va='bottom',fontsize=8)
 ax.text(0,7.4,'Encode update + old EF\nPersist pending bytes\nand staged residual',ha='center',va='top',fontsize=8,bbox={'facecolor':'#E6F0F5','edgecolor':'none','pad':3})
 arrow(0,1.55,6.2,'Authenticated envelope');arrow(1.55,3.1,5.4,'Register digest + context');arrow(3.1,1.55,4.6,'Receipt + bound readback')
 ax.text(1.55,4.1,'Accept contribution\nFinalize fixed participant set\nPersist resulting model',ha='center',va='top',fontsize=8,bbox={'facecolor':'#E9F2E9','edgecolor':'none','pad':3})
 arrow(1.55,0,2.5,'Authenticated incorporation receipt')
 ax.text(0,2.1,'Check freshness + binding\nCommit staged EF once',ha='center',va='top',fontsize=8,bbox={'facecolor':'#F9EEE6','edgecolor':'none','pad':3})
 arrow(0,1.55,.65,'Exact retry / state query');ax.text(2.35,.7,'Return persisted state;\nreuse existing registration',ha='center',va='center',fontsize=8)
 fig.savefig(P/'figures/figure3_upload_protocol.pdf');fig.savefig(P/'figures/figure3_upload_protocol.png',dpi=200)
