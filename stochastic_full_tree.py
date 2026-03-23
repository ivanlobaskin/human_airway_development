# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:46:34 2024

@author: ivanl

Stochastic full tree simulation
"""

#%% IMPORT LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import time
import math

rng = np.random.default_rng()

#%% FUNCTION DEFINITION

def jsd(p,q):
    m = 0.5*(p+q)
    return(st.entropy(m)-0.5*(st.entropy(p)+st.entropy(q)))

def bootstrap(data,samplesize,binss,n=1000):
    hists = np.zeros([n,len(binss)-1])
    for i in range(n):
        sample = rng.choice(data,samplesize)
        hists[i] = np.histogram(sample,bins=binss,density=True)[0]
    means = np.mean(hists,0)
    errs = np.std(hists,0)
    return(means,errs)

#%% IMPORT DATA

path1 = "C:\\Users\\ivanl\\OneDrive - University of Cambridge\\lung_exp_data\\full_lobes\\left_upper\\"
path2 = "EH4452-P9.4"

df1 = pd.read_csv(path1+path2+"\\tips.csv",header=None)
df2 = pd.read_csv(path1+path2+"\\tips2.csv",header=None) 
#df3 = pd.read_csv(path1+path2+"\\prune_10um\\tipwids.csv",header=None)
#df4 = pd.read_csv(path1+path2+"\\prune_10um\\tipwids2.csv",header=None) 

trim = [False,False]
if trim[0] :
    l = df1.to_numpy()[2,:]
    # w = df3.to_numpy()/10000
    w = np.ones(len(l))*np.mean(df1.to_numpy()[3,:])
    exp1 = np.array([li for i,li in enumerate(l) if li>w[i]])
else :
    exp1 = pd.to_numeric([l for i,l in enumerate(df1.T[2]) if df1.T[1][i]=='True'])
    # exp1 = pd.to_numeric([l for i,l in enumerate(df1.T[2])])

if trim[1] :
    l = df2.to_numpy()[2,:]
    w = df2.to_numpy()[2,:]
    exp2 = np.array([li for i,li in enumerate(l) if li>w[i]])
else :
    exp2 = pd.to_numeric([l for i,l in enumerate(df2.T[2]) if df2.T[1][i]=='True'])
    # exp2 = pd.to_numeric([l for i,l in enumerate(df2.T[2])])

#%% BOOTSTRAP
 
x_min = 10
x_max = 400
dx = 20
x_range = np.arange(x_min,x_max,dx)
    
exp1m, exp1e = bootstrap(exp1,len(exp1),x_range)
exp2m, exp2e = bootstrap(exp2,len(exp2),x_range)

#%% SIMULATION DEFINITION

def bud_rate_fun(c1,c2,c3,x):
    y = c1**-1/(1+np.exp(-(x-c2)/c3)) # Two tailed exponential
    # y = (1/c1)*(x/c2)**(c3)/(1+(x/c2)**c3) # Rate is a hill function
    # y = (c3/c2)*(x/c2)**(c3-1)/(1+(x/c2)**c3) # CDF is a hill function
    # if x < c2 :
    #     y = (1/c1)*(x/c2)**c3
    # else :
    #     y = 1/c1
    return(y)

def branch_rate_fun(c1,c2,c3,t):
    # y1 = (2*np.pi*c2**2)**(-0.5)*np.exp(-(t-c1)**2/(2*c2**2)) 
    # y2 = 0.5*(1+math.erf((t-c1)/(2**0.5*c2)))
    # y = y1/(1-y2)
    y = 1/c1
    # y = c1**-1/(1+np.exp(-(t-c2)/c3))
    # if t < c2 :
    #     y = (1/c1)*(t/c2)**c3
    # else :
    #     y = 1/c1
    return(y)

def initial_len_fun(c1,c2,t):
    # y = t
    y = 0
    # y = int(abs(rng.normal(c1,c2)))
    # y = int(rng.uniform(0,c1))
    # y = int(abs(rng.uniform(0,c1)+rng.normal(0,c2)))
    return(y)

def simulation(b,n_max,t_max):
    
    # Parameter order:
    # 0 x bud max; 1 x bud mid; 2 x bud growth 
    # 3 x branch a; 4 x branch b; 5 x branch c
    # 6 x initial a; 7 x initial b
     
    nbranches = 1
    lengths = [0]
    active = [1]
    timecreated = [0]
    edges = [[1,2]]
    initlens = []
    
    for t in range(t_max) :
        # lengths = [l+active[i] for i,l in enumerate(lengths)] 
        if nbranches > n_max :
            break
        
        for i in [i for i,a in enumerate(active) if a==1] :
            lengths[i] += 1
            bud_rate = bud_rate_fun(b[0],b[1],b[2],lengths[i]) 
            if rng.choice(2,p=[1-bud_rate,bud_rate]) :
                active[i] = 0 
                
                nbranches += 2
                lengths += [0,0]
                active += [2,1]
                timecreated += [t,t] 
                edges += [[i+2,nbranches],[i+2,nbranches+1]] 
                
        for i in [i for i,a in enumerate(active) if a==2] :
            branch_rate = branch_rate_fun(b[3],b[4],b[5],t-timecreated[i]) 
            if rng.choice(2,p=[1-branch_rate,branch_rate]) : 
                active[i] = 1 
                lengths[i] = initial_len_fun(b[6],b[7],t-timecreated[i])
                initlens += [lengths[i],lengths[i+1]]
                
    return(lengths,edges,initlens)

#%% RUN SIMULATION

t1 = time.time()
lengths,edges,initlens = simulation([40,90,15,1.,70.,3.,0.,0.],10000,10000)
t2 = time.time()

print(f"Done in : {t2-t1:.3f} s. Total branches : {len(lengths)}.")

# plt.hist(initlens[0::2]);

#%% TRIM + FIND TIPS

trimlength = 0
 
t1 = time.time()
while sum([l==0 for l in lengths]) :
    i = [i for i,l in enumerate(lengths) if l==0][0]
    j = edges[i][0]
    
    lengths.pop(i)
    edges.pop(i)
    
    i1 = [i for i,e in enumerate(edges) if e[1]==j][0]
    i2 = [i for i,e in enumerate(edges) if e[0]==j][0]
    
    if lengths[i2] :
        lengths += [lengths[i1]+lengths[i2]]
        lengths.pop(max(i1,i2))
        lengths.pop(min(i1,i2))
        
        edges += [[edges[i1][0],edges[i2][1]]]
        edges.pop(max(i1,i2))
        edges.pop(min(i1,i2))
    else : 
        lengths.pop(i2)
        edges.pop(i2)

ends = [0 for e in edges]
for i,e in enumerate(edges) :
    if not sum([e[1]==f[0] for f in edges]) :
        ends[i] = 1
        
ends2 = [0 for e in edges]
for e0 in [e[0] for i,e in enumerate(edges) if ends[i]] :
    j = [i for i,e in enumerate(edges) if e[1]==e0][0]
    ends2[j] = 1
 
    # + abs(rng.normal(30,20))
    # + abs(rng.normal(75.1,8.3))
tiplens = [l + rng.uniform(0,110) for i,l in enumerate(lengths) if ends[i] and l>trimlength]
tiplens2 = [l for i,l in enumerate(lengths) if ends2[i]]
t2 = time.time()

print(f"Done in : {t2-t1:.3f} s. Total branches : {len(lengths)}.")

#%% HISTOGRAMS

if trim[0] : 
    sim1 = np.histogram([t for t in tiplens if t>np.mean(df1.to_numpy()[3,:])],bins=x_range,density=True)[0]
else :
    sim1 = np.histogram(tiplens,bins=x_range,density=True)[0]
    
if trim[1] : 
    sim2 = np.histogram([t for t in tiplens2 if t>np.mean(df1.to_numpy()[3,:])],bins=x_range,density=True)[0]
else :
    sim2 = np.histogram(tiplens2,bins=x_range,density=True)[0]

def myerror(x,y,e):
    return plt.fill_between(x,y-e,y+e,alpha=0.5,step="pre",color='#ff7f0e');

# plt.suptitle(path2);

jsd1 = jsd(sim1,exp1m)
jsd2 = jsd(sim2,exp2m)

fig = plt.figure()
fig.set_size_inches((8,4));

plt.subplot(121);
plt.step(x_range[:-1],sim1);
plt.step(x_range[:-1],exp1m);
myerror(x_range[:-1],exp1m,exp1e)
plt.xlabel(r'Tip length ($\mu$m)');
plt.ylabel('Probability');
# plt.legend(["model","experiment"]);
plt.title(f'LogError : {np.log(jsd1):.3f}'); 
# plt.title('Tips');
#plt.title(f" params : {params}"); 

# plt.subplot(223);
# plt.step(x_range[:-1],sim1);
# plt.step(x_range[:-1],exp1m); 
# myerror(x_range[:-1],exp1m,exp1e) 
# plt.yscale("log");
 
plt.subplot(122);  
plt.step(x_range[:-1],sim2);
plt.step(x_range[:-1],exp2m); 
myerror(x_range[:-1],exp2m,exp2e)
plt.legend(["model","experiment"]);
plt.xlabel(r'Second segment length ($\mu$m)');
# plt.title('Second segments');
plt.title(f'LogError : {np.log(jsd2):.3f}');  

# plt.subplot(224);
# plt.step(x_range[:-1],sim2);
# plt.step(x_range[:-1],exp2m);
# myerror(x_range[:-1],exp2m,exp2e)
# plt.yscale("log");


#%% CDF + Survival function plot

def mycdf(x):
    return plt.hist(x,bins=200,cumulative='True',density='True',histtype='step',log='True');
def mysf(x):
    return plt.hist(x,bins=200,cumulative=-1,density='True',histtype='step',log='True');

fig = plt.figure()
fig.set_size_inches((6,4));

plt.subplot(221);
mycdf(tiplens);
mycdf(exp1);
plt.xlabel(r'Tip length ($\mu$m)');
plt.ylabel('CDF');
test1 = st.cramervonmises_2samp(exp1,tiplens)
plt.title(f'CvM = {test1.statistic:.3f}, p = {test1.pvalue:.3f}');

plt.subplot(223);
mysf(tiplens);
mysf(exp1);
plt.xlabel(r'Tip length ($\mu$m)');
plt.ylabel('SF');

plt.subplot(222);
mycdf(tiplens2);
mycdf(exp2);
plt.xlabel(r'Order 2 branch length ($\mu$m)');
plt.ylabel('CDF');
plt.legend(["model","experiment"]);
test2 = st.cramervonmises_2samp(exp2,tiplens2)
plt.title(f'CvM = {test2.statistic:.3f}, p = {test2.pvalue:.3f}');

plt.subplot(224);
mysf(tiplens2);
mysf(exp2);
plt.xlabel(r'Order 2 branch length ($\mu$m)');
plt.ylabel('SF');

#%% SAVE FILE TO CSV  

data = {"Vertex 1" : [e[0] for e in edges] , "Vertex 2" : [e[1] for e in edges] ,\
        "Lengths" : lengths, "Widths" : [rng.normal(70,10) for i in lengths] }

df = pd.DataFrame(data)
path = "C:\\Users\\ivanl\\OneDrive - University of Cambridge\\python\\"
df.to_csv(path+"simulation.csv",index=False)
#%%
mydist = st.fisk.fit(exp2)
