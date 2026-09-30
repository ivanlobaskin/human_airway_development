# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:27:07 2026

@author: ivanl
"""

#%% IMPORT LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import scipy.optimize as op
import time
from pathlib import Path

rng = np.random.default_rng()

#%% IMPORT DATA

data1 = Path(__file__).resolve().parent.parent / "data" / "tips.csv"
data2 = Path(__file__).resolve().parent.parent / "data" / "tips2.csv"

df1 = pd.read_csv(data1,header=None)
df2 = pd.read_csv(data2,header=None) 

include_surface = 1
include_bulk = 0

exp1 = pd.to_numeric([l for i,l in enumerate(df1.T[2]) if \
                      (df1.T[1][i]=='True' and include_surface) \
                      or (df1.T[1][i]=='False' and include_bulk) ])
exp2 = pd.to_numeric([l for i,l in enumerate(df2.T[2]) if \
                      (df2.T[1][i]=='True' and include_surface) \
                      or (df2.T[1][i]=='False' and include_bulk) ])

#%% SIMULATION DEFINITION 

# Budding rate as a function of current branch length x.
def bud_rate_fun(l_max,l_mid,l_gr,x):
    y = l_max**-1/(1+np.exp(-(x-l_mid)/l_gr)) # Two tailed exponential
    return(y)

# Delayed tip sprouting rate.
def branch_rate_fun(l_del,t):
    y = 1/l_del # Constant branch rate. This can be changed to a time-dependent branch rate.
    return(y)

# Distribution of tip bulb sizes (added after simulation).
def tip_size_fun(l_add,n=1):
    y = rng.uniform(0,l_add,n)
    return(y)

# Algorithm to generate random tree
def simulation(b, n_max, t_max, dt=1):
    
    # Parameters:
    # b : 0, l_max; 1, l_mid; 2, l_gr; 3, l_del
    # n_max : stop simulation when n_max branches have been created
    # t_max : stop simulation after t_max time steps
    # dt : time increment in each time step. Can be used to accelerate simulation,
    #      but may break model assumptions if increased too much.

    # Preallocate arrays
    nalloc = 2*n_max # This adds a buffer to prevent array overflow errors.
    
    lengths = np.zeros(nalloc)
    timecreated = np.zeros(nalloc, dtype=np.int32)
    
    # Keep edges as a numerical array internally rather than a list of lists.
    # This is converted back to the original format at the end.
    edges = np.zeros((nalloc, 2), dtype=np.int32)
    
    # Branch 0 is the initial growing tip
    nbranches = 1
    growing = np.array([0], dtype=np.int32)
    delayed = np.empty(0, dtype=np.int32)
    
    edges[0] = [1, 2]
    
    # Simulation
    for t in range(t_max):
        
        # A budding event always creates two branches, so don't create a pair
        # if doing so would exceed n_max.
        if nbranches + 2 > n_max:
            break
        
        # Growing tips
        
        if growing.size:
            
            # Elongate growing tips
            lengths[growing] += dt
            
            # Calculate budding probabilities for all growing tips 
            bud_rates = bud_rate_fun(b[0],b[1],b[2],lengths[growing])
            
            bud_probs = 1.0 - np.exp(-dt * bud_rates)
            
            # One random number per growing tip
            buds = rng.random(growing.size) < bud_probs
            
            budding = growing[buds]
            
            # Tips that did not bud remain active/growing
            growing = growing[~buds]
            
            # Create the two daughter branches for all budding tips            
            if budding.size:
                
                nnew = 2 * budding.size
                new_indices = np.arange(nbranches,nbranches+nnew,dtype=np.int32)
                timecreated[new_indices] = t
                
                # First daughter is delayed, second daughter is growing
                new_delayed = new_indices[0::2]
                new_growing = new_indices[1::2]
                
                
                edges[new_delayed, 0] = budding + 2
                edges[new_delayed, 1] = new_delayed + 2
                
                edges[new_growing, 0] = budding + 2
                edges[new_growing, 1] = new_growing + 2
                
                # Add new tips to their respective active sets
                delayed = np.concatenate((delayed, new_delayed))
                growing = np.concatenate((growing, new_growing))
                
                nbranches += nnew
        
        # Delayed tips
        if delayed.size:
            
            # Calculate ages and branching probabilities
            ages = t - timecreated[delayed]
            
            branch_rates = branch_rate_fun(b[3],ages)
            
            branch_probs = 1.0 - np.exp(-dt * branch_rates)
            
            # Determine which delayed tips sprout
            sprout = rng.random(delayed.size) < branch_probs
            
            # Move sprouted tips from delayed to growing
            if np.any(sprout):
                new_growing = delayed[sprout]
                
                growing = np.concatenate((growing, new_growing))
                delayed = delayed[~sprout]
    
    return (nbranches,lengths[:nbranches],edges[:nbranches])

# Algorithm to trim unsprouted buds and extract order 1 and 2 branches.
def trim_and_add(lengths, edges, l_add):
    
    # Remove zero-length branches and recursively reconnect their neighbours.    
    while np.any(lengths == 0):
        
        # First zero-length branch
        i = np.flatnonzero(lengths == 0)[0]
        
        # Parent node associated with this branch
        j = edges[i, 0]
        
        # Remove the zero-length branch
        lengths = np.delete(lengths, i)
        edges = np.delete(edges, i, axis=0)
        
        # Find branches connected to j
        i1 = np.flatnonzero(edges[:, 1] == j)[0]
        i2 = np.flatnonzero(edges[:, 0] == j)[0]
        
            
        # Combine the two branches
        new_length = lengths[i1] + lengths[i2]
        new_edge = np.array([edges[i1, 0], edges[i2, 1]],dtype=np.int32)
        
        # Delete both old branches
        lengths = np.delete(lengths, [i1, i2])
        edges = np.delete(edges, [i1, i2], axis=0)
        
        # Add combined branch
        lengths = np.append(lengths, new_length)
        edges = np.vstack((edges, new_edge))
    
    
    # Identify tips.

    # A branch is a tip if its end node does not occur as the start node
    # of another branch.
    
    end_nodes = edges[:, 1]
    start_nodes = edges[:, 0]
    
    tips = ~np.isin(end_nodes, start_nodes)
    
    # Add tip bulb size to tips
    lengths[tips] += tip_size_fun(l_add,np.sum(tips))
    
    # Identify order 2 branches (branches immediately preceding the tips).    
    tip_start_nodes = start_nodes[tips]
    
    tips2 = np.isin(end_nodes, tip_start_nodes)
    
    # Tip lengths    
    tiplens1 = lengths[tips] 
    tiplens2 = lengths[tips2]
    
    return (lengths,edges,tiplens1,tiplens2)

#%% FIND OPTIMAL PARAMETERS USING SIMULATED ANNEALING PACKAGE

# Simulation settings
n_max = int(5e3)
t_max = int(1e5)
dt = 2

# Test function.

# Takes set of model parameters as input.
# Runs simulation and extracts order 1 and 2 branch lengths.
# Calculates Cramer-von Mises test statistic for both, compared to sample data.
# Returns sum of squared test statistics (used as loss function in optimization).
def test_params(params):
    nbranches,lengths,edges = simulation(params[:4],n_max,t_max,dt)
    lengths,edges,tiplens1,tiplens2 = trim_and_add(lengths,edges,params[4])
    cvm1 = st.cramervonmises_2samp(exp1,tiplens1).statistic
    cvm2 = st.cramervonmises_2samp(exp2,tiplens2).statistic
    loss = cvm1**2+cvm2**2
    return(loss)


# Initial parameter guesses and range of parameters permitted in optimization.
# Might need to be adjusted for other samples.
params_init = np.array([
    25.,   # l_max
    90.,   # l_mid
    15.,   # l_gr
    100.,  # l_del
    65.    # l_add
    ])
params_range = np.array([
    (1,50),   # l_max
    (1,200),  # l_mid
    (1,50),   # l_gr
    (1,200),  # l_del
    (1,100)   # l_add
    ])

# Run optimization using 'dual_annealing' method from SciPy optimization package.
t1 = time.time()
result = op.dual_annealing(test_params,params_range,x0=params_init)
t2 = time.time()
print(f'Simulated Annealing complete in {t2-t1:.3f} s.')
print(f'Best parameters: \
      l_max = {result['x'][0]:.1f}, \
      l_mid = {result['x'][1]:.1f}, \
      l_gr = {result['x'][2]:.1f}, \
      l_del = {result['x'][3]:.1f}, \
      l_add = {result['x'][4]:.1f}')

#%% RUN SIMULATION AND SHOW RESULTS WITH BEST PARAMETER VALUES

n_max = int(1e4)
t_max = int(1e5)
dt = 2

sim_params = result['x'][:4]
trim_params = result['x'][4]

t1 = time.time()
nbranches,lengths,edges = simulation(sim_params,n_max,t_max,dt)
t2 = time.time()

print(f"Simulation done in : {t2-t1:.3f} s. Total branches : {len(lengths)}.")

t1 = time.time()
lengths,edges,tiplens1,tiplens2 = trim_and_add(lengths,edges,trim_params)
t2 = time.time()

print(f"Trimming done in : {t2-t1:.3f} s. Remaining branches : {len(lengths)}.")

test1 = st.cramervonmises_2samp(exp1,tiplens1)
print(f'Tips: Cramer-von Mises criterion = {test1.statistic:.3f}, p-value = {test1.pvalue:.3f}');

test2 = st.cramervonmises_2samp(exp2,tiplens2)
print(f'Order 2: Cramer-von Mises criterion = {test2.statistic:.3f}, p-value = {test2.pvalue:.3f}');

# PLOT RESULTS
# The plot types used are the cumulative distribution function (CDF)
# and the survival function (SF) for the order 1 and 2 branch lengths.
# The Cramer-von Mises criterion and its p-value are shown for each.
# Note that these plots are only to verify the goodness of fit.
# Code to generate figures is included in a Mathematica notebook.

def mypdf(ax,x):
    return ax.hist(x,bins=50,density=True,histtype='step');
def mycdf(ax,x):
    return ax.hist(x,bins=200,cumulative=True,density=True,histtype='step',log=True);
def mysf(ax,x):
    return ax.hist(x,bins=200,cumulative=-1,density=True,histtype='step',log=True);

fig, axs = plt.subplots(2,3,figsize=(8,4))
fig.tight_layout();

mypdf(axs[0,0],tiplens1);
mypdf(axs[0,0],exp1);
axs[0,0].set_xlabel(r'Tip length ($\mu$m)');
axs[0,0].set_ylabel('PDF');
axs[0,0].legend(["model","experiment"]);

mycdf(axs[0,1],tiplens1);
mycdf(axs[0,1],exp1);
axs[0,1].set_xlabel(r'Tip length ($\mu$m)');
axs[0,1].set_ylabel('CDF');

mysf(axs[0,2],tiplens1);
mysf(axs[0,2],exp1);
axs[0,2].set_xlabel(r'Tip length ($\mu$m)');
axs[0,2].set_ylabel('SF');

mypdf(axs[1,0],tiplens2);
mypdf(axs[1,0],exp2);
axs[1,0].set_xlabel(r'Order 2 branch length ($\mu$m)');
axs[1,0].set_ylabel('PDF');

mycdf(axs[1,1],tiplens2);
mycdf(axs[1,1],exp2);
axs[1,1].set_xlabel(r'Order 2 branch length ($\mu$m)');
axs[1,1].set_ylabel('CDF');

mysf(axs[1,2],tiplens2);
mysf(axs[1,2],exp2);
axs[1,2].set_xlabel(r'Order 2 branch length ($\mu$m)');
axs[1,2].set_ylabel('SF');

#%% EXPORT SIMULATION RESULT AS AN EDGE LIST

# A "Diameters" field is included so that the file can be processed
# using the same pipeline as the experimental data, although the simulations
# currently do not generate branch diameters.
data = {
        "Vertex 1" : edges[:,0] , 
        "Vertex 2" : edges[:,1] ,
        "Lengths" : lengths, 
        "Diameters" : np.ones(len(lengths)) ,
        "l_max" : result['x'][0] ,
        "l_mid" : result['x'][1] ,
        "l_gr" : result['x'][2] ,
        "l_del" : result['x'][3] ,
        "l_add" : result['x'][4]
        }

df = pd.DataFrame(data)

out_path = Path(__file__).resolve().parent.parent / "data" / "file_name.csv"
df.to_csv(out_path,index=False)
