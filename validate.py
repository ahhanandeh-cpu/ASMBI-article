import numpy as np, time, mpmath as mp
from math import comb, factorial
from scipy.integrate import quad
from scipy.stats import gamma as gd, multinomial
from itertools import product
from pe import GOLD, AKASH
rng=np.random.default_rng(7)
for fam,th in ((GOLD(4),1.2),(AKASH,1.5),(GOLD(2),1.0)):
    m=4
    # 1. order-statistic density vs direct
    errs=[]
    for i in range(1,m+1):
        for x in (0.2,1.0,3.0,7.0):
            S=fam.sf(x,th); d=m*comb(m-1,i-1)*(1-S)**(i-1)*S**(m-i)*fam.pdf(x,th)
            errs.append(abs(fam.os_pdf(x,th,m,i)-d)/max(d,1e-300))
    # integrates to one; moments
    e=[quad(lambda x: fam.os_pdf(x,th,m,i),0,np.inf)[0] for i in range(1,m+1)]
    mom=[fam.os_moments(th,m,i) for i in range(1,m+1)]
    num=[quad(lambda x: x*fam.os_pdf(x,th,m,i),0,np.inf)[0] for i in range(1,m+1)]
    print(fam.name, "max rel err os pdf %.1e"%max(errs), "integrals", np.round(e,12), "mean err %.1e"%max(abs(a[0]-b) for a,b in zip(mom,num)),
          "avg of os means - mu %.1e"%(np.mean([a[0] for a in mom])-fam.mean(th)))
# 2. SRS total: Talbot vs exact multinomial Erlang mixture
fam,th,n=GOLD(3),0.6,6
w=fam.weights(th); lam=1/th
def exact_srs_cdf(t):
    tot=0.0
    for cnt in product(range(n+1),repeat=3):
        if sum(cnt)!=n: continue
        pr=multinomial.pmf(cnt,n,w); sh=sum((j+1)*c for j,c in enumerate(cnt))
        tot+=pr*gd.cdf(t,sh,scale=1/lam)
    return tot
for t in (1.5,3,6):
    a=fam.cdf_T(t,th,('srs',n)); b=exact_srs_cdf(t)
    print("SRS cdf talbot %.15f exact %.15f diff %.1e"%(a,b,abs(a-b)))
# 3. RSS total: talbot vs dehoog vs MC, perfect and imperfect
fam,th,m,r=GOLD(2),1.0,3,3
lamb=0.4; Pm=(1-lamb)*np.eye(m)+lamb/m
N=300000
pool=fam.rvs(N*r*m*m,th,rng).reshape(N,r,m,m); srt=np.sort(pool,axis=3)
Tperf=sum(srt[:,:,i,i] for i in range(m)).sum(1)
# imperfect: judged rank i -> true rank j with prob P[i,j]
jt=np.array([[rng.choice(m,p=Pm[i]) for i in range(m)] for _ in range(1)])
J=np.stack([rng.choice(m,size=(N,r),p=Pm[i]) for i in range(m)],axis=2)   # N,r,m
Timp=np.take_along_axis(srt,J[...,None],axis=3)[...,0].sum((1,2))
for t in (6,9,12):
    L=fam.lt_total(th,('rss',m,r,None))
    a=fam.cdf_T(t,th,('rss',m,r,None)); b=float(mp.invertlaplace(lambda s:L(s)/s,t,method='dehoog'))
    c=fam.cdf_T(t,th,('rss',m,r,Pm))
    print("RSS t=%g talbot %.12f dehoog %.12f MC %.4f | imperfect exact %.4f MC %.4f"%(t,a,b,(Tperf<=t).mean(),c,(Timp<=t).mean()))
t0=time.time(); fam.cdf_T(9,th,('rss',m,r,None)); print("one cdf eval %.3fs"%(time.time()-t0))
t0=time.time(); ci=fam.exact_ci(9.0,('rss',m,r,None)); print("CI",ci,"%.2fs"%(time.time()-t0))
