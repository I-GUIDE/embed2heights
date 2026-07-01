"""Export per-tile missing-building masks (v2 density-gap detector) to runs/missing_masks/<core>.npy
(uint8, label-native HxW). Only flagged tiles get a file; absent file = no masking for that tile.
These mark pixels where building footprint labels were (we believe) deleted; training zeroes the
presence/seg loss there so the model isn't penalized for predicting building (analogous to ndsm_hole)."""
import os,sys,glob,re; import numpy as np, rasterio
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,'.'); from core.metrics import build_label_map
import os as _os; _D=_os.environ.get("DATA_ROOT","./data")+"/train"; LAB=_D+"/labels"; AE=_D+"/alphaearth_emb"
LM=build_label_map(LAB)
GT,H_THR,SIG_THR=0.10,3.0,0.60; R,GAP_THR,AREA_MIN,H_GUARD=6,0.25,80,2.0
st=ndimage.generate_binary_structure(2,2)
OUT='runs/missing_masks'; os.makedirs(OUT,exist_ok=True)
AEI={}
for p in glob.glob(os.path.join(AE,'*.tif')):
    m=re.search(r'(\d{4}_[A-Z]{2})',os.path.basename(p))
    if m: AEI[m.group(1)]=p
def boxblur(x,r): return ndimage.uniform_filter(x.astype(np.float32),size=2*r+1,mode='nearest')
rng=np.random.default_rng(0); Xb,Xv=[],[]
for t in rng.choice(sorted(LM),60,replace=False):
    with rasterio.open(LM[t]) as s: lab=s.read().astype(np.float32)
    with rasterio.open(AEI[t]) as s: e=np.nan_to_num(s.read().astype(np.float32),nan=0.0)
    H=min(lab.shape[1],e.shape[1]);W=min(lab.shape[2],e.shape[2]);lab=lab[:,:H,:W];e=e[:,:H,:W]
    ev=e.reshape(e.shape[0],-1).T; eu=ev/(np.linalg.norm(ev,axis=1,keepdims=True)+1e-6)
    bi=np.flatnonzero((lab[0]>0.5).ravel()); vi=np.flatnonzero((lab[1]>0.5).ravel())
    if len(bi): Xb.append(eu[rng.choice(bi,min(300,len(bi)),replace=False)])
    if len(vi): Xv.append(eu[rng.choice(vi,min(300,len(vi)),replace=False)])
Xb,Xv=np.vstack(Xb),np.vstack(Xv)
clf=LogisticRegression(max_iter=300).fit(np.vstack([Xb,Xv]),np.r_[np.ones(len(Xb)),np.zeros(len(Xv))])
W_=clf.coef_[0].astype(np.float32); B_=np.float32(clf.intercept_[0])
def pbuild(e):
    ev=e.reshape(e.shape[0],-1).T; eu=ev/(np.linalg.norm(ev,axis=1,keepdims=True)+1e-6)
    return (1/(1+np.exp(-(eu@W_+B_)))).reshape(e.shape[1],e.shape[2])
def detect(lab,e):
    bld,veg,wat,hgt=lab; E=((hgt>H_THR)&(pbuild(e)>SIG_THR)).astype(np.float32)
    gap=boxblur(E,R)-boxblur((bld>GT).astype(np.float32),R)
    flag=(gap>GAP_THR)&(bld<=GT)&(veg<=GT)&(wat<=GT)&(hgt>H_GUARD)
    flag=ndimage.binary_closing(flag,structure=st,iterations=1)
    lblc,nl=ndimage.label(flag,st)
    if not nl: return np.zeros_like(flag,bool)
    sz=ndimage.sum(np.ones_like(lblc),lblc,range(1,nl+1)); keep=np.flatnonzero(sz>=AREA_MIN)+1
    return np.isin(lblc,keep) if len(keep) else np.zeros_like(flag,bool)
nsaved=tot=0
for i,t in enumerate(sorted(LM)):
    if t not in AEI: continue
    with rasterio.open(LM[t]) as s: lab=s.read().astype(np.float32); Lh,Lw=lab.shape[1],lab.shape[2]
    with rasterio.open(AEI[t]) as s: e=np.nan_to_num(s.read().astype(np.float32),nan=0.0)
    H=min(lab.shape[1],e.shape[1]);W=min(lab.shape[2],e.shape[2])
    m=detect(lab[:,:H,:W],e[:,:H,:W])
    if m.sum()>=AREA_MIN:
        full=np.zeros((Lh,Lw),np.uint8); full[:H,:W]=m.astype(np.uint8)   # label-native shape
        np.save(os.path.join(OUT,f'{t}.npy'),full); nsaved+=1; tot+=int(m.sum())
    if (i+1)%500==0: print(f'  {i+1} scanned, {nsaved} masks',flush=True)
print(f'saved {nsaved} masks ({tot:,} px) to {OUT}/')
