"""Fair re-eval using the OFFICIAL binary_iou (empty->1.0 convention, matches cov0p10_eval):
building IoU on fold0 val, EXCLUDING flagged missing-building pixels for BOTH models."""
import os,sys,glob,json; import numpy as np, rasterio
sys.path.insert(0,'.'); from core.metrics import CH_BUILDING, build_label_map, binary_iou
LABELS='/u/dingqi2/workspace/esa/data/train/labels'; GT_COV=0.10
BASE='runs/xf095_2stage_softbin_covgt10_s0_f0/predictions'
DELM='runs/xf095_2stage_softbin_covgt10_delmask_s0_f0/predictions'
MASKD='runs/missing_masks'
VAL=set(json.load(open('splits/group_code_5fold_seed42/fold_0/split.json'))['val'])
LM=build_label_map(LABELS); grid=np.round(np.arange(0.20,0.951,0.025),3)
def core_of(p): b=os.path.basename(p)[:-4]; return '_'.join(b.split('_')[:2])
acc={'base':{'dirty':{t:[] for t in grid},'clean':{t:[] for t in grid}},
     'delm':{'dirty':{t:[] for t in grid},'clean':{t:[] for t in grid}}}
nfl=0; flag_bld={'base':[],'delm':[]}; ntiles=0
for pf in sorted(glob.glob(os.path.join(BASE,'*.npy'))):
    core=core_of(pf)
    if core not in VAL or core not in LM: continue
    pb=np.load(pf); pd_=np.load(os.path.join(DELM,os.path.basename(pf)))
    with rasterio.open(LM[core]) as s: lab=s.read().astype(np.float32)
    h=min(pb.shape[1],pd_.shape[1],lab.shape[1]); w=min(pb.shape[2],pd_.shape[2],lab.shape[2])
    pb=pb[:, :h,:w]; pd_=pd_[:, :h,:w]; lab=lab[:, :h,:w]
    gt=lab[CH_BUILDING]>GT_COV
    flagged=np.zeros((h,w),bool); mp=os.path.join(MASKD,f'{core}.npy')
    if os.path.exists(mp):
        mm=np.load(mp); fh=min(mm.shape[0],h); fw=min(mm.shape[1],w); flagged[:fh,:fw]=mm[:fh,:fw].astype(bool); nfl+=1
    keep=~flagged; ntiles+=1
    for t in grid:
        bb=pb[CH_BUILDING]>t; dd=pd_[CH_BUILDING]>t
        acc['base']['dirty'][t].append(binary_iou(bb,gt));        acc['base']['clean'][t].append(binary_iou(bb[keep],gt[keep]))
        acc['delm']['dirty'][t].append(binary_iou(dd,gt));        acc['delm']['clean'][t].append(binary_iou(dd[keep],gt[keep]))
    if flagged.any():
        flag_bld['base'].append((pb[CH_BUILDING][flagged]>0.5).mean()); flag_bld['delm'].append((pd_[CH_BUILDING][flagged]>0.5).mean())
def best(d): bt=max(grid,key=lambda t:np.mean(d[t])); return bt,float(np.mean(d[bt]))
print(f"fold0 val: {ntiles} tiles, {nfl} flagged  (official binary_iou, empty->1.0)\n")
res={}
for m,name in [('base','BASELINE (no-mask train)'),('delm','DELMASK (mask train)')]:
    bd,id_=best(acc[m]['dirty']); bc,ic=best(acc[m]['clean']); res[m]=(id_,ic)
    print(f"{name}\n   DIRTY (all px vs gappy GT):  thr {bd:.3f} -> bld IoU {id_:.4f}\n   CLEAN (exclude flagged px):  thr {bc:.3f} -> bld IoU {ic:.4f}")
print(f"\nbuilding predicted INSIDE flagged regions (frac px>0.5): baseline {np.mean(flag_bld['base']):.3f}  delmask {np.mean(flag_bld['delm']):.3f}")
print(f"\n=== delmask - baseline ===")
print(f"  DIRTY (gappy, what I reported): {res['delm'][0]-res['base'][0]:+.4f}  ({res['base'][0]:.4f} -> {res['delm'][0]:.4f})")
print(f"  CLEAN (fair, flagged excluded): {res['delm'][1]-res['base'][1]:+.4f}  ({res['base'][1]:.4f} -> {res['delm'][1]:.4f})")
