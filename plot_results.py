"""Reproduce the released quantitative figures from main_table.csv."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PKG=Path(__file__).resolve().parent
MODELS=['idefics3','qwen9b','qwen','internvl35','pixtral12b','llava15']
NAMES=['Idefics3-8B','Qwen3.5-9B','Qwen3.5-27B','InternVL3.5-8B','Pixtral-12B','LLaVA-1.5-7B']
with (PKG/'results/main_table.csv').open(encoding='utf-8-sig',newline='') as handle: table=list(csv.DictReader(handle))
for row in table:
    for key in row:
        if key not in {'model','scoring','cohort','scope','donor_category'} and row[key]: row[key]=float(row[key])
    row['pairs']=int(row['pairs']);row['targets']=int(row['targets'])
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,'svg.fonttype':'none','pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10,5.1),sharey=True)
for ax,scope,title in zip(axes,['all','human_majority_model_reliable'],['All 500 target images','Human majority + model checks']):
    for i,model in enumerate(MODELS):
        r=next(r for r in table if r['model']==model and r['scoring']=='format_tolerant' and r['cohort']=='new_500' and r['scope']==scope)
        t,n=[100*r[k]/r['pairs'] for k in ('T_follow','N-near_follow')]
        ax.plot([n,t],[i,i],color='#a5a5a5',lw=2)
        ax.scatter(t,i,color='#b74b43',s=50,label='Covering target' if i==0 else None)
        ax.scatter(n,i,color='#277b87',s=50,marker='s',label='Near target' if i==0 else None)
        ax.text(61,i,f"n={r['pairs']}; images={r['targets']}",fontsize=7,va='center')
    ax.set_yticks(range(6),NAMES);ax.set_xlim(-2,89);ax.set_xticks([0,20,40,60]);ax.set_xlabel('Donor-category answers (%)')
    ax.set_title(title);ax.grid(axis='x',alpha=.18)
axes[0].invert_yaxis();axes[0].legend(loc='lower left',bbox_to_anchor=(0,1.08),ncol=2,fontsize=8)
fig.suptitle('Spatial placement changes directed donor answers',y=1.02,fontsize=13)
fig.tight_layout()
directory=PKG/'figures';directory.mkdir(exist_ok=True)
fig.savefig(directory/'main_comparison.png',dpi=300,bbox_inches='tight')
fig.savefig(directory/'main_comparison.pdf',bbox_inches='tight')
fig.savefig(directory/'main_comparison.svg',bbox_inches='tight')
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,5),sharey=True)
colors=['#43836d','#b74b43','#96938c','#d2b464','#866999'];labels=['Target correct','Donor answer','Other error','UNKNOWN','Ambiguous']
for ax,condition,title in zip(axes,['T','N-near'],['Covering target','Near target']):
    left=np.zeros(6)
    for suffix,color,label in zip(['correct','follow','other','unknown','ambiguous'],colors,labels):
        counts=np.array([next(r for r in table if r['model']==m and r['scoring']=='format_tolerant' and r['cohort']=='new_500' and r['scope']=='all')[condition+'_'+suffix]/10 for m in MODELS])
        ax.barh(range(6),counts,left=left,color=color,label=label,height=.65);left+=counts
    ax.set_title(title);ax.set_xlim(0,100);ax.set_yticks(range(6),NAMES);ax.set_xlabel('Answers (%)')
axes[0].invert_yaxis();axes[1].legend(loc='upper center',bbox_to_anchor=(-.15,-.15),ncol=3,fontsize=8)
fig.tight_layout()
fig.savefig(directory/'answer_types.png',dpi=300,bbox_inches='tight')
fig.savefig(directory/'answer_types.pdf',bbox_inches='tight')
fig.savefig(directory/'answer_types.svg',bbox_inches='tight')
plt.close(fig)
