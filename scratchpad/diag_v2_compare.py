"""Compare v1 (per-pixel+blob50) vs v2 (density-gap) on the two complaint tiles."""
import os,sys,glob; import numpy as np, rasterio
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,'scratchpad'); sys.path.insert(0,'.')
from missing_building_detector_v2 import LM,ae_of,pbuild,detect_v2,st,H_THR,SIG_THR,GT,AUC,R,GAP_THR,AREA_MIN
print(f'signature AUC {AUC:.3f}  R={R} gap>{GAP_THR} area>={AREA_MIN}')
def detect_v1(lab,e):
    bld,veg,wat,hgt=lab
    cand=(bld==0)&(veg==0)&(wat==0)&(hgt>H_THR)&(pbuild(e)>SIG_THR)
    lblc,nl=ndimage.label(cand,st)
    if not nl: return np.zeros_like(cand)
    sz=ndimage.sum(np.ones_like(lblc),lblc,range(1,nl+1)); keep=np.flatnonzero(sz>=50)+1
    return np.isin(lblc,keep) if len(keep) else np.zeros_like(cand)
TILES=['1597_GD','1439_KE']
fig,ax=plt.subplots(len(TILES),5,figsize=(17,3.4*len(TILES)))
for r,t in enumerate(TILES):
    with rasterio.open(LM[t]) as s: lab=s.read().astype(np.float32)
    with rasterio.open(ae_of(t)) as s: e=np.nan_to_num(s.read().astype(np.float32),nan=0.0)
    H=min(lab.shape[1],e.shape[1]);W=min(lab.shape[2],e.shape[2]);lab=lab[:,:H,:W];e=e[:,:H,:W]
    bld,veg,wat,hgt=lab
    cov=np.zeros((H,W,3)); cov[...,0]=bld>GT; cov[...,1]=veg>GT; cov[...,2]=wat>GT
    m1=detect_v1(lab,e); m2,gap=detect_v2(lab,e,return_gap=True)
    ax[r,0].imshow(cov); ax[r,0].set_ylabel(t,fontsize=11)
    ax[r,1].imshow(gap,cmap='RdBu_r',vmin=-0.6,vmax=0.6)
    rep1=cov*0.45; rep1[m1]=[1,1,0]; ax[r,2].imshow(np.clip(rep1,0,1)); ax[r,2].set_xlabel(f'v1 {int(m1.sum())}px',fontsize=9)
    rep2=cov*0.45; rep2[m2]=[1,1,0]; ax[r,3].imshow(np.clip(rep2,0,1)); ax[r,3].set_xlabel(f'v2 {int(m2.sum())}px',fontsize=9)
    # recovered by v2 not v1 = cyan; lost = red
    diff=cov*0.45; diff[m1&~m2]=[1,0,0]; diff[m2&~m1]=[0,1,1]; diff[m1&m2]=[1,1,0]
    ax[r,4].imshow(np.clip(diff,0,1)); ax[r,4].set_xlabel('yellow=both cyan=v2-only red=v1-only',fontsize=8)
    for j in range(5): ax[r,j].set_xticks([]); ax[r,j].set_yticks([])
for j,ti in enumerate(['cov>0.10 RGB','density gap E-L','v1 overlay','v2 overlay','v2 vs v1']):
    ax[0,j].set_title(ti,fontsize=10)
plt.tight_layout(); plt.savefig('scratchpad/diag_v2_compare.png',dpi=110,bbox_inches='tight'); print('saved scratchpad/diag_v2_compare.png')
