#!/usr/bin/env python3
"""Pair‑width correlation & calibration trend (compact).

Reads *xgboost_standard_bootstrap_va_test_predictions.csv*,
prints decile‑trend correlations, writes:

    width_pairwise_correlation.csv   # columns in raw‑data names
    width_distribution_grid_test.png
    decile_trend_grid_test.png

No other CSVs are produced."""

import warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------ paths ----
BASE = Path("/mnt/c/Users/bebek/Desktop/Master/Thesis/OUTPUTS")
IN_DIR  = BASE / "va_bootstraps_xgboost"
OUT_DIR = BASE / "width_correlation_analysis"; OUT_DIR.mkdir(exist_ok=True,parents=True)
CSV = IN_DIR / "xgboost_standard_bootstrap_va_test_predictions.csv"

# ---------------------------------------------------------------- utilities --
def tcol(df):
    for c in df.columns:
        if c.startswith('true_') or c.lower() in ('churn','exited','target','y','label'):
            return c
    raise ValueError('target column not found')

def decile_corr(df,w,y):
    df['bin']=pd.qcut(df[w],10,labels=False,duplicates='drop')
    s=df.groupby('bin')[y].mean()*100
    r,p=pearsonr(range(len(s)),s) if len(s)>1 else (0,1)
    return s,r,p

disp = lambda s:s.replace('_',' ').title()

# ---------------------------------------------------------------- main -------
df=pd.read_csv(CSV); y=tcol(df)
wcols=[c for c in df.columns if c.endswith('_width') and c!='pi_width']  # keep raw names
print("width columns:",wcols)

# ---- decile trend
trend={}
for w in wcols:
    s,r,p=decile_corr(df[[w,y]].dropna().copy(),w,y)
    trend[w]=(s,r,p)
    print(f"{'✅' if p<.05 else '❌'} {w}: r={r:.3f}, p={p:.4f}")

# ---- pairwise correlation csv
rows=[]
for i,a in enumerate(wcols):
    for b in wcols[i+1:]:
        tmp=df[[a,b]].dropna()
        pr,pp=pearsonr(tmp[a],tmp[b]); sr,sp=spearmanr(tmp[a],tmp[b])
        rows.append({'width_a':a,'width_b':b,
                     'pearson':pr,'pearson_p':pp,
                     'spearman':sr,'spearman_p':sp,'n':len(tmp)})
pd.DataFrame(rows).to_csv(OUT_DIR/'width_pairwise_correlation.csv',index=False)

# ---- distribution grid
fig,ax=plt.subplots(2,2,figsize=(12,10)); ax=ax.flatten()
for i,w in enumerate(wcols[:4]):
    churn=df[df[y]==1][w].dropna(); keep=df[df[y]==0][w].dropna()
    ax[i].hist(keep,30,density=True,alpha=.6,color='lightblue',label='Retained')
    ax[i].hist(churn,30,density=True,alpha=.6,color='steelblue',label='Churned')
    ax[i].set_title(disp(w)); ax[i].grid(alpha=.3); ax[i].legend()
for i in range(len(wcols),4): ax[i].set_visible(False)
plt.suptitle('Width Distribution by Churn Status - Test'); plt.tight_layout()
plt.savefig(OUT_DIR/'width_distribution_grid_test.png',dpi=300); plt.close()

# ---- decile trend grid
fig,ax=plt.subplots(2,2,figsize=(12,10)); ax=ax.flatten()
colors=['skyblue','lightgreen','lightcoral','lightsalmon']
for i,w in enumerate(wcols[:4]):
    s,r,p=trend[w]; bars=ax[i].bar(s.index,s.values,color=colors[i],edgecolor='k',alpha=.7)
    z=np.polyfit(s.index,s.values,1); ax[i].plot(s.index,np.poly1d(z)(s.index),'r--')
    ax[i].set_title(disp(w)); ax[i].grid(alpha=.3)
    ax[i].text(.02,.93,f"r={r:.3f}\np={p:.4f}",transform=ax[i].transAxes,
               bbox=dict(boxstyle='round',facecolor='w',alpha=.8))
for i in range(len(wcols),4): ax[i].set_visible(False)
plt.suptitle('Churn Rate by Width Decile - Test'); plt.tight_layout()
plt.savefig(OUT_DIR/'decile_trend_grid_test.png',dpi=300); plt.close()

print(f"\nFiles saved to {OUT_DIR}")
