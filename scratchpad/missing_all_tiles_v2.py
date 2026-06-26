"""Run v2 density-gap detector on ALL tiles -> ranked list + montage. Fast: prebuilt index, vectorized signature."""
import os,sys,glob,re; import numpy as np, rasterio
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,'.'); from core.metrics import build_label_map
LAB='/u/dingqi2/workspace/esa/data/train/labels'; AE='/u/dingqi2/workspace/esa/data/train/alphaearth_emb'
LM=build_label_map(LAB)
GT,H_THR,SIG_THR=0.10,3.0,0.60; R,GAP_THR,AREA_MIN=6,0.25,80; H_GUARD=2.0
st=ndimage.generate_binary_structure(2,2)
# prebuilt tile->AE path index (no per-tile glob)
AEI={}
for p in glob.glob(os.path.join(AE,'*.tif')):
    m=re.search(r'(\d{4}_[A-Z]{2})',os.path.basename(p))
    if m: AEI[m.group(1)]=p
def boxblur(x,r): return ndimage.uniform_filter(x.astype(np.float32),size=2*r+1,mode='nearest')
# signature fit
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
AUC=roc_auc_score(np.r_[np.ones(len(Xb)),np.zeros(len(Xv))],clf.decision_function(np.vstack([Xb,Xv])))
W_=clf.coef_[0].astype(np.float32); B_=np.float32(clf.intercept_[0])
print(f'AUC {AUC:.3f}  R={R} gap>{GAP_THR} area>={AREA_MIN}',flush=True)
def pbuild(e):
    ev=e.reshape(e.shape[0],-1).T; eu=ev/(np.linalg.norm(ev,axis=1,keepdims=True)+1e-6)
    z=eu@W_+B_; return (1/(1+np.exp(-z))).reshape(e.shape[1],e.shape[2])
def detect(lab,e):
    bld,veg,wat,hgt=lab
    E=((hgt>H_THR)&(pbuild(e)>SIG_THR)).astype(np.float32)
    gap=boxblur(E,R)-boxblur((bld>GT).astype(np.float32),R)
    flag=(gap>GAP_THR)&(bld<=GT)&(veg<=GT)&(wat<=GT)&(hgt>H_GUARD)
    flag=ndimage.binary_closing(flag,structure=st,iterations=1)
    lblc,nl=ndimage.label(flag,st)
    if not nl: return np.zeros_like(flag)
    sz=ndimage.sum(np.ones_like(lblc),lblc,range(1,nl+1)); keep=np.flatnonzero(sz>=AREA_MIN)+1
    return np.isin(lblc,keep) if len(keep) else np.zeros_like(flag)
allt=sorted(LM); hits=[]
for i,t in enumerate(allt):
    if t not in AEI: continue
    with rasterio.open(LM[t]) as s: lab=s.read().astype(np.float32)
    with rasterio.open(AEI[t]) as s: e=np.nan_to_num(s.read().astype(np.float32),nan=0.0)
    H=min(lab.shape[1],e.shape[1]);W=min(lab.shape[2],e.shape[2]);lab=lab[:,:H,:W];e=e[:,:H,:W]
    m=detect(lab,e)
    if m.sum()>=AREA_MIN:
        cov=np.zeros((H,W,3)); cov[...,0]=lab[0]>GT; cov[...,1]=lab[1]>GT; cov[...,2]=lab[2]>GT
        hits.append((t,int(m.sum()),m,cov))
    if (i+1)%300==0: print(f'  {i+1}/{len(allt)} scanned, {len(hits)} flagged',flush=True)
hits.sort(key=lambda x:-x[1]); tot=sum(h[1] for h in hits)
print(f'\nALL {len(allt)} tiles: {len(hits)} flagged ({100*len(hits)/len(allt):.1f}%); total flagged px {tot:,}')
print('top 20:',[(t,p) for t,p,_,_ in hits[:20]])
with open('scratchpad/missing_v2_ranked.csv','w') as f:
    f.write('tile,flagged_px\n')
    for t,p,_,_ in hits: f.write(f'{t},{p}\n')
print('wrote scratchpad/missing_v2_ranked.csv')
TOP=hits[:64]; N=len(TOP); ncol=8; nrow=int(np.ceil(N/ncol))
fig,ax=plt.subplots(nrow,ncol,figsize=(ncol*2.0,nrow*2.0)); ax=np.atleast_2d(ax)
for idx in range(nrow*ncol):
    a=ax[idx//ncol,idx%ncol]; a.axis('off')
    if idx<N:
        t,px,m,cov=TOP[idx]; rep=cov*0.45; rep[m]=[1,1,0]
        a.imshow(np.clip(rep,0,1)); a.set_title(f'{t} {px}px',fontsize=6)
plt.suptitle(f'v2 density-gap: top {N} of {len(hits)} flagged (missing-bld YELLOW)',fontsize=11)
plt.tight_layout(); plt.savefig('scratchpad/missing_all_tiles_v2.png',dpi=120,bbox_inches='tight')
print('saved scratchpad/missing_all_tiles_v2.png')
