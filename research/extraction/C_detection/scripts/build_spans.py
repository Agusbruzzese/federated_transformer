import sys; sys.path.insert(0,'.')
import os; sys.path.insert(0,os.path.dirname(__file__)); import cio, detector as D, numpy as np, datetime as dt, json
R=cio.runs()
SPANS=[ # (name, run, start, category)
 ("r1_wed05_0745",0,dt.datetime(2025,2,5,7,45),"wk_rush_am"),
 ("r1_wed05_1230",0,dt.datetime(2025,2,5,12,30),"wk_midday"),
 ("r1_wed05_1715",0,dt.datetime(2025,2,5,17,15),"wk_evening"),
 ("r1_thu06_0230",0,dt.datetime(2025,2,6,2,30),"wk_night"),
 ("r1_fri07_0800",0,dt.datetime(2025,2,7,8,0),"wk_rush_am"),
 ("r1_sat08_1000",0,dt.datetime(2025,2,8,10,0),"we_day"),
 ("r1_sun09_0300",0,dt.datetime(2025,2,9,3,0),"we_night"),
 ("r1_sun09_1600",0,dt.datetime(2025,2,9,16,0),"we_day"),
 ("r2_thu13_0800",1,dt.datetime(2025,2,13,8,0),"wk_rush_am"),
 ("r2_thu13_1800",1,dt.datetime(2025,2,13,18,0),"wk_evening"),
 ("r3_wed19_0810",2,dt.datetime(2025,2,19,8,10),"wk_rush_am"),
 ("r3_tue18_2200",2,dt.datetime(2025,2,18,22,0),"wk_night"),
 ("r4_tue25_1300",3,dt.datetime(2025,2,25,13,0),"wk_midday"),
 ("r4_wed26_0400",3,dt.datetime(2025,2,26,4,0),"wk_night"),
 # ---- holdout (Run 3; PROTOCOL.md) ----
 ("r3_wed19_1230",2,dt.datetime(2025,2,19,12,30),"wk_midday"),
 ("r3_wed19_1715",2,dt.datetime(2025,2,19,17,15),"wk_evening"),
 ("r3_thu20_0300",2,dt.datetime(2025,2,20,3,0),"wk_night"),
 ("r3_thu20_0750",2,dt.datetime(2025,2,20,7,50),"wk_rush_am"),
 # added after the first holdout hand-check count (73 candidates < pre-registered 200): more Run-3 traffic
 ("r3_tue18_1730",2,dt.datetime(2025,2,18,17,30),"wk_evening"),
 ("r3_wed19_0730",2,dt.datetime(2025,2,19,7,30),"wk_rush_am"),
 ("r3_wed19_1600",2,dt.datetime(2025,2,19,16,0),"wk_evening"),
 ("r3_thu20_1200",2,dt.datetime(2025,2,20,12,0),"wk_midday"),
]
def load(r,i0,n):
    """stitched raw + per-sample chunk id + wall-clock epoch seconds (filename start + sample offset/FS_RAW)"""
    x,t,c=cio.load_chunks(r,i0,n)
    idx={k[0]:k for k in cio.index(r)}
    wall=np.empty(len(x)); 
    for cid in np.unique(c):
        m=np.where(c==cid)[0]; wall[m]=idx[cid][1].timestamp()+np.arange(len(m))/D.FS_RAW
    return x,t,c,wall
if __name__=='__main__':
    meta=[]
    for name,ri,w,cat in SPANS:
        r=R[ri]; i0=cio.find_chunk(r,w-dt.timedelta(minutes=2)); n=int(24*60/41.4)+2
        x,t,c,wall=load(r,i0,n)
        t50,y50,idx,drop=D.front_end(x,t)   # slot clock seconds, dropouts re-inserted
        _,slot_of_raw,_=D.insert_dropouts(x,t); ridx=np.clip(np.searchsorted(slot_of_raw,idx),0,len(x)-1)
        np.savez_compressed(f'data/span_{name}.npz',t=t50,y=y50,chunk=c[ridx],time=t[ridx],wall=wall[ridx],drop=drop)
        e0=np.searchsorted(wall[ridx],w.timestamp()); e1=np.searchsorted(wall[ridx],w.timestamp()+1200)
        meta.append(dict(name=name,split=('holdout' if ri==2 else 'tune'),run=ri,start=str(w),cat=cat,i0=i0,n=n,e0=int(e0),e1=int(e1),dow=w.strftime('%a'),hour=w.hour+w.minute/60))
        print(name,len(y50),e0,e1,round((e1-e0)/D.FS,1),'dropout frac',round(drop.mean(),4),'n dropouts',int((np.diff(t)>1.5*np.median(np.diff(t))).sum()))
    json.dump(meta,open('data/spans.json','w'),indent=1)
