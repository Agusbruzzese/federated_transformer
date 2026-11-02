import pandas as pd, numpy as np, os
HERE=os.path.dirname(os.path.abspath(__file__))
R=pd.read_csv(os.path.join(HERE,'recompute_scores.csv'),dtype={'run':str})
out=[]
for (pr,fo),g in R.groupby(['probe','folds']):
    M=g.groupby(['run','rep','score']).value.mean()
    for sc in ['polarity','lobes','duration']:
        for comp in ['tfinit','cnn_oracle','cnn01','cnninit','pca64','raw']:
            ds=[]
            for r in ['0204','0212','0218','0224']:
                if (r,'tf',sc) not in M: continue
                if comp=='cnn_oracle': b=max(M[(r,'cnn01',sc)],M[(r,'cnn02',sc)])
                elif (r,comp,sc) in M: b=M[(r,comp,sc)]
                else: continue
                ds.append(M[(r,'tf',sc)]-b)
            if len(ds)==4: out.append(dict(probe=pr,folds=fo,score=sc,comp=comp,mean=np.mean(ds),runs_pos=sum(d>0 for d in ds),per_run=' '.join(f'{d:+.3f}' for d in ds)))
O=pd.DataFrame(out); O.to_csv(os.path.join(HERE,'recompute_deltas.csv'),index=False)
pd.set_option('display.width',200); print(O.round(3).to_string(index=False))
