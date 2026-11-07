"""Study A: exact finite-sample MSE (RSS-ME, RSS median-unbiased) vs SRS-MLE.  Study B: exact coverage of Wald(ME)."""
import numpy as np, json, sys
from scipy.integrate import quad
from scipy.optimize import brentq
from multiprocessing import Pool
from pe import GOLD, AKASH
Z=1.959964
FAMS={"Lindley":(GOLD(2),1.0),"Sujatha":(GOLD(3),0.6),"Amarendra":(GOLD(4),1.2),"Akash":(AKASH,1.5)}

def dmu(fam,th,h=1e-6): return (fam.mean(th*(1+h))-fam.mean(th*(1-h)))/(2*th*h)

def tgrid_moments(fam,th,design,g):
    """E g(T) and E (g(T)-th)^2 by Simpson's rule on the exact density of T (401 nodes).
    Accuracy is checked against the closed-form mass (1) and mean (n*mu)."""
    from scipy.integrate import simpson
    n=fam.n_of(design); mu=n*fam.mean(th); sd=np.sqrt(n*fam.var(th))
    t=np.linspace(max(1e-9,mu-8*sd),mu+12*sd,401)
    f=np.array([fam.pdf_T(x,th,design) for x in t])
    mass=simpson(f,x=t); mean=simpson(t*f,x=t)
    assert abs(mass-1)<1e-7 and abs(mean/mu-1)<1e-7, (mass,mean,mu)
    gv=np.array([g(x) for x in t])
    return simpson(gv*f,x=t), simpson((gv-th)**2*f,x=t)

import os
def job(args):
    name,n,m=args
    fn=f"cache/ab_{name}_{n}_{m}.json"
    if os.path.exists(fn): return json.load(open(fn))
    o=_job(args); json.dump(o,open(fn,"w"),default=float); return o
def _job(args):
    name,n,m=args
    fam,th=FAMS[name]
    srs=('srs',n); rss=('rss',m,n//m,None)
    me=lambda t: fam.mu_inv(t/n)
    e_s,mse_s=tgrid_moments(fam,th,srs,me)
    e_r,mse_r=tgrid_moments(fam,th,rss,me)
    # Wald(ME) exact coverage, RSS: se = sqrt(sigbar2/n)/mu'
    def se_rss(tt):
        v=np.mean([fam.os_moments(tt,m,i)[1] for i in range(1,m+1)])
        return np.sqrt(v/n)/abs(dmu(fam,tt))
    def se_srs(tt): return np.sqrt(fam.var(tt)/n)/abs(dmu(fam,tt))
    out={"name":name,"n":n,"m":m,"theta":th,"bias_srs":e_s-th,"mse_srs":mse_s,"bias_rss":e_r-th,"mse_rss":mse_r,"re_exact":mse_s/mse_r}
    for lab,des,se in (("srs",srs,se_srs),("rss",rss,se_rss)):
        t0=n*fam.mean(th)
        h=lambda t: abs(me(t)-th)-Z*se(me(t))
        sd=np.sqrt(n*fam.var(th))
        a=max(1e-3*t0,t0-12*sd)
        t1=brentq(h,a,t0) if h(a)>0 else None
        t2=brentq(h,t0,t0+25*sd)
        F1=fam.cdf_T(t1,th,des) if t1 else 0.0; F2=fam.cdf_T(t2,th,des)
        out[f"wald_{lab}"]=dict(cover=F2-F1,T_high_miss=1-F2,T_low_miss=F1)
    return out

if __name__=="__main__":
    jobs=[(nm,n,3) for nm in FAMS for n in (6,9,15,30)]+[("Lindley",60,m) for m in (2,3,4,5)]
    with Pool(2) as p: R=p.map(job,jobs)
    json.dump(R,open("study_ab.json","w"),indent=1,default=float)
    for o in R:
        print("%-9s n=%3d m=%d  bias S %.4f R %.4f | MSE S %.5f R %.5f RE %.3f | Wald cov S %.4f R %.4f (R miss T-low %.4f T-high %.4f)"%(
            o['name'],o['n'],o['m'],o['bias_srs'],o['bias_rss'],o['mse_srs'],o['mse_rss'],o['re_exact'],
            o['wald_srs']['cover'],o['wald_rss']['cover'],o['wald_rss']['T_low_miss'],o['wald_rss']['T_high_miss']))
