"""Region-level missing-building detector (v2).

WHY v1 failed: v1 = per-pixel (noclass & height>3 & p_build>0.6) then drop blobs<50px.
The AND of three noisy criteria is salt-and-pepper; genuine under-labeled REGIONS get
broken into 1-2px fragments by interspersed labeled pixels + height/signature holes, so
the 50px component filter deletes them (e.g. 1439_KE: 5781 candidate px -> only 1485 kept).

v2 idea: a missing region is where BUILDING EVIDENCE is dense but LABELS are sparse.
Work on smoothed DENSITIES, not per-pixel ANDs:
  E(x) = building evidence (LABEL-INDEPENDENT): height>H_THR & p_build>SIG
  L(x) = labeled building: bld>0.10
  gap  = boxblur(E, R) - boxblur(L, R)        # high where evidence >> labels nearby
  flag = (gap > GAP_THR) & (bld<=0.10)        # only surface actually-unlabeled pixels
  then morphological close + keep regions >= AREA_MIN
Robust to salt-and-pepper because it integrates evidence over a neighborhood.
"""
import os,sys,glob,json
import numpy as np, rasterio
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'.'); from core.metrics import build_label_map
LAB='/u/dingqi2/workspace/esa/data/train/labels'; AE='/u/dingqi2/workspace/esa/data/train/alphaearth_emb'
LM=build_label_map(LAB)
GT,H_THR,SIG_THR = 0.10, 3.0, 0.60
R, GAP_THR, AREA_MIN = 6, 0.25, 80     # neighborhood radius, density-gap thr, region size
H_GUARD = 2.0                          # per-pixel elevation guard: a flagged px must itself be raised,
                                       # killing flat-ground bleed from the neighborhood gap test
st=ndimage.generate_binary_structure(2,2)

def ae_of(t): return glob.glob(os.path.join(AE,f'*{t}*.tif'))[0]
def fit_sig():
    rng=np.random.default_rng(0); Xb,Xv=[],[]
    for t in rng.choice(sorted(LM),60,replace=False):
        with rasterio.open(LM[t]) as s: lab=s.read().astype(np.float32)
        with rasterio.open(ae_of(t)) as s: e=np.nan_to_num(s.read().astype(np.float32),nan=0.0)
        H=min(lab.shape[1],e.shape[1]);W=min(lab.shape[2],e.shape[2]);lab=lab[:,:H,:W];e=e[:,:H,:W]
        ev=e.reshape(e.shape[0],-1).T; eu=ev/(np.linalg.norm(ev,axis=1,keepdims=True)+1e-6)
        bi=np.flatnonzero((lab[0]>0.5).ravel()); vi=np.flatnonzero((lab[1]>0.5).ravel())
        if len(bi): Xb.append(eu[rng.choice(bi,min(300,len(bi)),replace=False)])
        if len(vi): Xv.append(eu[rng.choice(vi,min(300,len(vi)),replace=False)])
    Xb,Xv=np.vstack(Xb),np.vstack(Xv)
    clf=LogisticRegression(max_iter=300).fit(np.vstack([Xb,Xv]),np.r_[np.ones(len(Xb)),np.zeros(len(Xv))])
    auc=roc_auc_score(np.r_[np.ones(len(Xb)),np.zeros(len(Xv))],clf.decision_function(np.vstack([Xb,Xv])))
    return clf,auc
clf,AUC=fit_sig()
def pbuild(e):
    ev=e.reshape(e.shape[0],-1).T; eu=ev/(np.linalg.norm(ev,axis=1,keepdims=True)+1e-6)
    return clf.predict_proba(eu)[:,1].reshape(e.shape[1],e.shape[2])
def boxblur(x,r): return ndimage.uniform_filter(x.astype(np.float32),size=2*r+1,mode='nearest')

def detect_v2(lab,e,return_gap=False):
    bld,veg,wat,hgt=lab
    E=((hgt>H_THR)&(pbuild(e)>SIG_THR)).astype(np.float32)
    L=(bld>GT).astype(np.float32)
    gap=boxblur(E,R)-boxblur(L,R)
    flag=(gap>GAP_THR)&(bld<=GT)&(veg<=GT)&(wat<=GT)&(hgt>H_GUARD)
    flag=ndimage.binary_closing(flag,structure=st,iterations=1)
    lblc,nl=ndimage.label(flag,st)
    if nl:
        sizes=ndimage.sum(np.ones_like(lblc),lblc,range(1,nl+1)); keep=np.flatnonzero(sizes>=AREA_MIN)+1
        out=np.isin(lblc,keep) if len(keep) else np.zeros_like(flag)
    else: out=np.zeros_like(flag)
    return (out,gap) if return_gap else out
