#!/usr/bin/env python3
"""Simple (compact) width‑analysis.
   • Reads xgboost_standard_bootstrap_va_test_predictions.csv
   • Prints summary stats, pairwise correlations.
   • Exports
        ‑ width_summary.csv
        ‑ width_pairwise_correlation.csv
        ‑ width_distributions_grid.png      (2×2 hist+kde)
        ‑ width_distributions_original.png  (1×N raw hists)
        ‑ churn_lift_*.png                  (per‑width lift)
        ‑ churn_lift_combined.png
        ‑ width_scatter_plots.png           (≤6 pair scatter)
        ‑ width_correlations.png            (heat‑map corr)
        ‑ churner_distribution_by_uncertainty.csv
"""

import os, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, gaussian_kde
warnings.filterwarnings('ignore')

# ---------- paths ------------------------------------------------------------
BASE = Path("/mnt/c/Users/bebek/Desktop/Master/Thesis/OUTPUTS")
IN_DIR   = BASE / "va_bootstraps_xgboost"
OUT_DIR  = BASE / "width_analysis_simple"; OUT_DIR.mkdir(exist_ok=True,parents=True)
CSV      = IN_DIR / "xgboost_standard_bootstrap_va_test_predictions.csv"

# ---------- helpers ----------------------------------------------------------
def target_col(df):
    for c in df:
        if c.startswith('true_') or c.lower() in ('churn','exited','target','y','label'):
            return c
    raise ValueError('target column missing')

def display(name):  # prettify raw width col for plots
    return name.replace('_',' ').title()

# ---------- load -------------------------------------------------------------
df = pd.read_csv(CSV)
y  = target_col(df)
wcols = [c for c in df if c.endswith('_width') and c!='pi_width']
print("Width columns:", wcols)

# ---------- summary stats ----------------------------------------------------
rows=[]
for c in wcols:
    s=df[c].dropna()
    rows.append({'width_type':c,'count':len(s),'mean':s.mean(),'median':s.median(),
                 'std':s.std(),'min':s.min(),'max':s.max(),
                 'q25':np.percentile(s,25),'q75':np.percentile(s,75),'skew':s.skew()})
    print(f"{c}: mean={s.mean():.6f}  sd={s.std():.6f}  skew={s.skew():.3f}")
pd.DataFrame(rows).to_csv(OUT_DIR/'width_summary.csv',index=False)

# ---------- pairwise correlation --------------------------------------------
pairs=[]
for i,a in enumerate(wcols):
    for b in wcols[i+1:]:
        tmp=df[[a,b]].dropna()
        pr,pp=pearsonr(tmp[a],tmp[b]); sr,sp=spearmanr(tmp[a],tmp[b])
        pairs.append({'width_a':a,'width_b':b,
                      'pearson':pr,'pearson_p':pp,
                      'spearman':sr,'spearman_p':sp,'n':len(tmp)})
pd.DataFrame(pairs).to_csv(OUT_DIR/'width_pairwise_correlation.csv',index=False)

# ---------- distribution grid (2×2) -----------------------------------------
fig,ax=plt.subplots(2,2,figsize=(12,10)); ax=ax.flatten()
for i,w in enumerate(wcols[:4]):
    s=df[w].dropna()
    ax[i].hist(s,50,density=True,alpha=.7,color='skyblue',edgecolor='k')
    kde=gaussian_kde(s); x=np.linspace(s.min(),s.max(),300)
    ax[i].plot(x,kde(x),'r-',lw=2); ax[i].set_title(display(w)); ax[i].grid(alpha=.3)
for i in range(len(wcols),4): ax[i].set_visible(False)
plt.tight_layout(); plt.savefig(OUT_DIR/'width_distributions_grid.png',dpi=300); plt.close()

# ---------- raw hists --------------------------------------------------------
fig,ax=plt.subplots(1,len(wcols),figsize=(6*len(wcols),5))
if len(wcols)==1: ax=[ax]
for a,w in zip(ax,wcols):
    a.hist(df[w].dropna(),50,edgecolor='k',alpha=.7); a.set_title(display(w)); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT_DIR/'width_distributions_original.png',dpi=300); plt.close()

# ---------- churn‑lift -------------------------------------------------------
def lift_plot(d,w):
    t=d[[w,y]].dropna(); t['bin']=pd.qcut(t[w],10,labels=False,duplicates='drop')
    rate=t.groupby('bin')[y].mean()*100
    plt.figure(figsize=(8,5))
    plt.bar(rate.index,rate,color='steelblue'); plt.plot(rate.index,rate,'ro-',lw=2)
    plt.title(f'Churn % vs Uncertainty ({display(w)})'); plt.grid(alpha=.3)
    plt.xlabel('Percentile bin'); plt.ylabel('Churn %')
    plt.tight_layout(); plt.savefig(OUT_DIR/f'churn_lift_{w}.png',dpi=300); plt.close()
    return rate

rates={w:lift_plot(df,w) for w in wcols}

# combined
plt.figure(figsize=(10,7))
for m,(w,rate) in zip('os^D',rates.items()):
    plt.plot(rate.index,rate,marker=m,lw=2,label=display(w))
plt.legend(); plt.grid(alpha=.3)
plt.xlabel('Percentile bin'); plt.ylabel('Churn %')
plt.title('Churn Rate by Uncertainty Percentile')
plt.tight_layout(); plt.savefig(OUT_DIR/'churn_lift_combined.png',dpi=300); plt.close()

# ---------- scatter (<=6) & heatmap ------------------------------------------
pairs=list(pairs)[:6]
if pairs:
    fig,ax=plt.subplots(2,3,figsize=(15,10)); ax=ax.flatten()
    for k,pair in enumerate(pairs):
        a,b=pair['width_a'],pair['width_b']
        tmp=df[[a,b]].dropna()
        ax[k].scatter(tmp[a],tmp[b],s=10,alpha=.5)
        z=np.polyfit(tmp[a],tmp[b],1); ax[k].plot(tmp[a],np.poly1d(z)(tmp[a]),'r--')
        ax[k].text(.05,.9,f"r={pair['pearson']:.3f}",transform=ax[k].transAxes)
        ax[k].set_xlabel(display(a)); ax[k].set_ylabel(display(b)); ax[k].grid(alpha=.3)
    for k in range(len(pairs),6): ax[k].set_visible(False)
    plt.suptitle('Width Scatter Plots'); plt.tight_layout()
    plt.savefig(OUT_DIR/'width_scatter_plots.png',dpi=300); plt.close()

sns.heatmap(df[wcols].corr(),annot=True,cmap='coolwarm',square=True,cbar_kws={'shrink':.8})
plt.title('Width Correlation Matrix'); plt.tight_layout()
plt.savefig(OUT_DIR/'width_correlations.png',dpi=300); plt.close()

# ---------- churn distribution csv -------------------------------------------
rows=[]
for w in wcols:
    v=df[[w,y]].dropna()[w]; c=df[y].loc[v.index]
    for p in (50,75,90,95):
        thr=np.percentile(v,p); hi=v>thr; lo=~hi
        rows.append({'width_type':w,'percentile':p,'threshold':thr,
                     'total_high_uncertainty':hi.sum(),
                     'churners_high_uncertainty':c[hi].sum(),
                     'churn_pct_high_uncertainty':c[hi].mean()*100 if hi.sum() else 0,
                     'total_low_uncertainty':lo.sum(),
                     'churners_low_uncertainty':c[lo].sum(),
                     'churn_pct_low_uncertainty':c[lo].mean()*100 if lo.sum() else 0})
pd.DataFrame(rows).to_csv(OUT_DIR/'churner_distribution_by_uncertainty.csv',index=False)

print(f"\n✅ Outputs written to {OUT_DIR}")
