"""
Spectral Clustering Optimisation Pipeline
==========================================
Finds the optimal number of clusters using Spectral Clustering with full graph
construction parameter search, evaluated by the reduced asymmetric NMI.

Usage:
    pip install -r requirements.txt
    python run_pipeline.py --data data.csv --output dashboard.html
"""
import argparse, ast, json, os, warnings
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from clustering_mi import normalized_mutual_information as nmi_reduced
warnings.filterwarnings("ignore")

def load_data(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    emb = np.array([ast.literal_eval(e) for e in df["embedding"]])
    und = df["understanding_level"].values
    grd = df["final_grade_sum"].values
    print(f"  {len(df)} students, {emb.shape[1]}-dim embeddings")
    return df, emb, und, grd

def spec_nmi(X, und, k, affinity, gamma=None, n_neighbors=None):
    p = {"n_clusters":k,"affinity":affinity,"assign_labels":"discretize","random_state":42}
    if affinity=="rbf": p["gamma"]=gamma
    else: p["n_neighbors"]=n_neighbors
    try:
        cl = SpectralClustering(**p).fit_predict(X)
        sz = [int((cl==i).sum()) for i in range(k)]
        if min(sz)<5: return -0.1,0.0,cl,sz
        nr = float(nmi_reduced(cl,und,variation="reduced",normalization="second"))
        ns = float(normalized_mutual_info_score(und,cl))
        return nr,ns,cl,sz
    except: return -0.1,0.0,None,[]

def run_search(emb, und, grd):
    gs = StandardScaler().fit_transform(grd.reshape(-1,1))
    pca_dims=[3,5,7,10,15,20]; std_opts=[False,True]; gws=[0,1,3,5]; ks=[3,4,5]
    gps = [{"a":"rbf","g":g} for g in [0.005,0.01,0.02,0.05,0.1,0.2,0.5]] + \
          [{"a":"nearest_neighbors","n":n} for n in [5,7,10,15,20,30]]
    total=len(pca_dims)*len(std_opts)*len(gws)*len(ks)*len(gps)
    print(f"  {total} configs..."); recs=[]; cnt=0
    for nc in pca_dims:
        pca=PCA(n_components=nc,random_state=42); Xp=pca.fit_transform(emb); ve=sum(pca.explained_variance_ratio_)
        for ds in std_opts:
            for gw in gws:
                Xf=np.hstack([Xp,gs*gw]) if gw>0 else Xp.copy()
                if ds: Xf=StandardScaler().fit_transform(Xf)
                for gp in gps:
                    for k in ks:
                        cnt+=1; nr,ns,_,sz=spec_nmi(Xf,und,k,gp["a"],gp.get("g"),gp.get("n"))
                        recs.append({"pca":nc,"std":ds,"grade_w":gw,"k":k,"affinity":gp["a"],
                            "gamma":gp.get("g"),"n_neighbors":gp.get("n"),
                            "nmi_red_A":round(nr,4),"nmi_sym":round(ns,4),"var_expl":round(ve,3),"sizes":sz})
                        if cnt%500==0: print(f"    {cnt}/{total}...")
    res=pd.DataFrame(recs).sort_values("nmi_red_A",ascending=False)
    print(f"  Best: NMI={res.iloc[0]['nmi_red_A']}"); return res

def compute_curves(emb, und, grd, kr):
    gs=StandardScaler().fit_transform(grd.reshape(-1,1))
    cfgs=[
        {"l":"NN=10, PCA(10) std","p":10,"s":True,"gw":0,"a":"nearest_neighbors","n":10,"g":None},
        {"l":"NN=5, PCA(10) std+gw3","p":10,"s":True,"gw":3,"a":"nearest_neighbors","n":5,"g":None},
        {"l":"NN=7, PCA(10) std","p":10,"s":True,"gw":0,"a":"nearest_neighbors","n":7,"g":None},
        {"l":"RBF \u03b3=0.5, PCA(10) std","p":10,"s":True,"gw":0,"a":"rbf","n":None,"g":0.5},
        {"l":"RBF \u03b3=0.01, PCA(15) std","p":15,"s":True,"gw":0,"a":"rbf","n":None,"g":0.01},
    ]
    res={}
    for c in cfgs:
        print(f"  Curve: {c['l']}...")
        pca=PCA(n_components=c["p"],random_state=42); X=pca.fit_transform(emb)
        if c["gw"]>0: X=np.hstack([X,gs*c["gw"]])
        if c["s"]: X=StandardScaler().fit_transform(X)
        curve=[round(spec_nmi(X,und,k,c["a"],c["g"],c["n"])[0],4) for k in kr]
        print(f"    Best k={kr[np.argmax(curve)]}, NMI={max(curve):.4f}")
        res[c["l"]]=curve
    return res

def run_optimal(emb, und):
    pca=PCA(n_components=10,random_state=42); Xp=pca.fit_transform(emb)
    Xs=StandardScaler().fit_transform(Xp); ve=sum(pca.explained_variance_ratio_)
    cl=SpectralClustering(n_clusters=4,affinity="nearest_neighbors",n_neighbors=10,assign_labels="discretize",random_state=42).fit_predict(Xs)
    nr=float(nmi_reduced(cl,und,variation="reduced",normalization="second"))
    ns=float(normalized_mutual_info_score(und,cl))
    ct=pd.crosstab(cl,und); chi2,pv,dof,exp=chi2_contingency(ct); n=len(und)
    cv=float(np.sqrt(chi2/(n*(min(ct.shape)-1))))
    stats={"chi2":round(chi2,2),"p_value":float(pv),"dof":dof,"cramers_v":round(cv,4),
           "correlated":1 if pv<0.05 else 0,"observed":ct.values.tolist(),
           "expected":np.round(exp,1).tolist(),
           "contributions":np.round((ct.values-exp)**2/exp,2).tolist(),
           "row_labels":[f"Cluster {i}" for i in ct.index.tolist()],
           "col_labels":[f"Level {i}" for i in ct.columns.tolist()],
           "row_totals":ct.sum(axis=1).tolist(),"col_totals":ct.sum(axis=0).tolist(),"grand_total":n}
    opt={"pca":10,"std":True,"affinity":"nearest_neighbors","n_neighbors":10,"k":4,
         "nmi_red_A":round(nr,4),"nmi_sym":round(ns,4),"var_explained":round(ve,3)}
    return cl,Xs,stats,opt

def run_umap(Xs,cl,und):
    import umap.umap_ as u; print("  UMAP...")
    X2=u.UMAP(n_components=2,random_state=42).fit_transform(Xs)
    return [{"x":round(float(X2[i,0]),3),"y":round(float(X2[i,1]),3),"cluster":int(cl[i]),"level":int(und[i])} for i in range(len(cl))]

def run_grade(emb,und,grd,kr):
    gs=StandardScaler().fit_transform(grd.reshape(-1,1))
    pca=PCA(n_components=10,random_state=42); Xp=pca.fit_transform(emb)
    Xn=StandardScaler().fit_transform(Xp)
    Xg=StandardScaler().fit_transform(np.hstack([Xp,gs*3]))
    nn=[round(spec_nmi(Xn,und,k,"nearest_neighbors",n_neighbors=10)[0],4) for k in kr]
    ng=[round(spec_nmi(Xg,und,k,"nearest_neighbors",n_neighbors=5)[0],4) for k in kr]
    return {"k_range":list(kr),"nmi_no_grade":nn,"nmi_with_grade":ng}

def search_summary(df):
    rows=[]
    for k in [3,4,5]:
        b=df[df["k"]==k].iloc[0]
        a=f"NN={int(b['n_neighbors'])}" if b["affinity"]=="nearest_neighbors" else f"RBF \u03b3={b['gamma']}"
        rows.append({"config":f"PCA({b['pca']}), {'std' if b['std'] else 'raw'}, {a}"+
            (f", gw={b['grade_w']}" if b["grade_w"]>0 else ""),
            "k":int(k),"nmi_red":float(b["nmi_red_A"]),"nmi_sym":float(b["nmi_sym"]),
            "note":"Best overall" if k==4 else f"Best for k={k}"})
    return rows

def gen_html(cr,kr,stats,opt,umap,ge,ss,path):
    print(f"Writing {path}...")
    o=opt; cl=list(cr.keys()); cc=[cr[l] for l in cl]; colors=["#534AB7","#D85A30","#1D9E75","#888780","#D4537E"]
    D=json.dumps({"cfgLabels":cl,"cfgColors":colors,"cfgCurves":cc,"kRange":list(kr),
        "stats":stats,"optimal":opt,"umap":umap,"gradeEffect":ge,"searchSummary":ss})
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"dashboard_template.html")) as f: tpl=f.read()
    html=tpl.replace("__DATA_JSON__",D).replace("__PCA__",str(o["pca"])).replace("__NN__",str(o["n_neighbors"])).replace("__K__",str(o["k"])).replace("__NMI__",str(o["nmi_red_A"])).replace("__VAR__",f"{o['var_explained']:.1%}")
    with open(path,"w") as f: f.write(html)
    print(f"  {os.path.getsize(path)/1024:.0f} KB")

def main():
    pa=argparse.ArgumentParser(); pa.add_argument("--data",default="data.csv"); pa.add_argument("--output",default="dashboard.html")
    a=pa.parse_args()
    df,emb,und,grd=load_data(a.data); kr=list(range(2,11))
    print("\n[1/6] Search..."); sdf=run_search(emb,und,grd); ss=search_summary(sdf)
    print("\n[2/6] Curves..."); cr=compute_curves(emb,und,grd,kr)
    print("\n[3/6] Optimal..."); cl,Xs,stats,opt=run_optimal(emb,und)
    print("\n[4/6] UMAP..."); um=run_umap(Xs,cl,und)
    print("\n[5/6] Grade..."); ge=run_grade(emb,und,grd,kr)
    print("\n[6/6] Dashboard..."); gen_html(cr,kr,stats,opt,um,ge,ss,a.output)
    sdf.head(50).to_csv(os.path.splitext(a.output)[0]+"_search.csv",index=False)
    print(f"\nDone! Open {a.output}")

if __name__=="__main__": main()
