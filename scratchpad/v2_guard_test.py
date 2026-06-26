import os,sys,glob,re; import numpy as np, rasterio
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,'scratchpad'); sys.path.insert(0,'.')
from missing_building_detector_v2 import LM,pbuild,GT,H_THR,SIG_THR,R,GAP_THR,AREA_MIN,st
AE='/u/dingqi2/workspace/esa/data/train/alphaearth_emb'; AEI={}
for p in glob.glob(os.path.join(AE,'*.tif')):
    m=re.search(r'(\d{4}_[A-Z]{2})',os.path.basename(p))
    if m: AEI[m.group(1)]=p
def boxblur(x,r): return ndimage.uniform_filter(x.astype(np.float32),size=2*r+1,mode='nearest')
def detect(lab,e,guard=None):
    bld,veg,wat,hgt=lab; E=((hgt>H_THR)&(pbuild(e)>SIG_THR)).astype(np.float32)
    gap=boxblur(E,R)-boxblur((bld>GT).astype(np.float32),R)
    flag=(gap>GAP_THR)&(bld<=GT)&(veg<=GT)&(wat<=GT)
    if guard is not None: flag=flag&(hgt>guard)        # per-pixel elevation guard
    flag=ndimage.binary_closing(flag,structure=st,iterations=1)
    lblc,nl=ndimage.label(flag,st)
    if not nl: return np.zeros_like(flag)
    sz=ndimage.sum(np.ones_like(lblc),lblc,range(1,nl+1)); keep=np.flatnonzero(sz>=AREA_MIN)+1
    return np.isin(lblc,keep) if len(keep) else np.zeros_like(flag)
TILES=['1597_GD','1439_KE','1939_PE','0995_PQ','1448_KE','0256_IQ']
fig,ax=plt.subplots(len(TILES),3,figsize=(10,3.2*len(TILES)))
for r,t in enumerate(TILES):
    with rasterio.open(LM[t]) as s: lab=s.read().astype(np.float32)
    with rasterio.open(AEI[t]) as s: e=np.nan_to_num(s.read().astype(np.float32),nan=0.0)
    H=min(lab.shape[1],e.shape[1]);W=min(lab.shape[2],e.shape[2]);lab=lab[:,:H,:W];e=e[:,:H,:W]
    bld,veg,wat,hgt=lab
    cov=np.zeros((H,W,3)); cov[...,0]=bld>GT; cov[...,1]=veg>GT; cov[...,2]=wat>GT
    m0=detect(lab,e,guard=None); mg=detect(lab,e,guard=2.0)
    ax[r,0].imshow(np.clip(hgt/8,0,1),cmap='inferno'); ax[r,0].set_ylabel(t,fontsize=9)
    rep=cov*0.45; rep[m0]=[1,1,0]; ax[r,1].imshow(np.clip(rep,0,1)); ax[r,1].set_xlabel(f'no guard {int(m0.sum())}px',fontsize=8)
    rep=cov*0.45; rep[mg]=[1,1,0]; ax[r,2].imshow(np.clip(rep,0,1)); ax[r,2].set_xlabel(f'guard>2m {int(mg.sum())}px',fontsize=8)
    for j in range(3): ax[r,j].set_xticks([]); ax[r,j].set_yticks([])
for j,ti in enumerate(['height (vmax=8m)','v2 no guard','v2 +height-guard>2m']): ax[0,j].set_title(ti,fontsize=10)
plt.tight_layout(); plt.savefig('scratchpad/v2_guard_test.png',dpi=110,bbox_inches='tight'); print('saved')
