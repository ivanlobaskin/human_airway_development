# OVERVIEW

This repository contains the code needed to perform data analysis on lung network files to extract relevant data and generate the graphs reported in the main and supplementary figures.
Also included is the code to run simulations of the model of network growth and the model of branching dynamics. <br><br>
The data folder contains data from one sample (EH4452, left upper lobe, pcw9.4) to run a demo of the analysis script and to show how models were fitted to the data.
It also contains the full processed data needed to generate the graphs for the figures. 
Lastly, some saved data from simulation runs are included.

# SYSTEM REQUIREMENTS

Wolfram Mathematica 14.3 installed. 
Some parts of the code may not execute correctly in earlier versions. <br><br>

Python 3.12.4 installed. 
Some parts of the code may not execute correctly in earlier versions.
Python libraries used: NumPy, MatPlotLib, Pandas, SciPy, Time, PathLib. <br><br>

Any OS that can run Mathematica and Python is sufficient.
No non-standard hardware is required.

# INSTALLATION GUIDE

1. Download and install Wolfram Mathematica 14.3 according to the instructions found [here](https://reference.wolfram.com/language/tutorial/InstallingWolfram.html).

2. Download and install Python 3.12 from the official website [here](https://www.python.org/downloads/) or another implementation.

Download and installation times may depend on download speed.
Note: Wolfram software may require a licence to download and install.

# SUMMARY OF CODE

## 1. Data analysis (lungs_data_analysis.nb; Mathematica notebook)

This notebook imports a network file (FULL_clean_edge_list.dat) and the corresponding node vertex coordinates (FULL_node_positions.dat) and extracts data used for analysis and for generating figures. <br><br>
The cells from the first two sections ("Import file and define graph object" and "Function definitions") must be run first.
Otherwise, cells can be run in any order.
Each subsection indicates which figure panel the corresponding output was used in.
The output is mostly printed in table form.
Execution of cells should be instantaneous.

## 2. Figure generation (lungs_figures.nb; Mathematica notebook) 

This notebook contains the code used to generate graphs for figures.
It imports the processed data from the corresponding files (lungs_fig_X.xlsx or lungs_supp_fig_X.xlsx).<br><br>
Run the first cell first.
Otherwise, sections can be run in any order.
In each section, cells must be executed sequentially.
Each section indicates which figure panel the corresponding graph is in.
The graphs are generated in the notebook and can be saved or exported.
Graph formatting may differ from the presentation in the figure.
Where appropriate, sections also include cells that extract reported key statistics (such as fit parameters, $R^2$ values).
Execution should be instantaneous.

## 3. Model of branching dynamics (lungs_model_of_branching_dynamics.py; Python code)

This code runs a simulation of the model of branching dynamics. <br><br>
Sections should be run sequentially.
The code will generate some plots to assess the goodness of the fit (these were not used in the figures) and export a csv file containing a network from a simulation with the best-fit parameters.
The parameter optimization code should take around half an hour to run on a standard desktop. All other sections should be instantaneous.

## 4. Model of network growth (lungs_model_of_network_growth.nb; Mathematica notebook)

This notebook runs a simulation of the network growth model. <br><br>

First, run "Import sample data".

1. To reproduce results comparing to a single sample, do one of the following: <br>
i. Run all cells in "Run new simulation". <br>
ii. Run "Import saved simulation data". <br>
In either case, then run "Convert simulation length scale and expansion rate to experiment equivalents".
Then, cells from "Extract data (single sample)" can be run in any order.

2. To reproduce results comparing to multiple samples, do one of the following: <br>
i. Run "Run new simulation/Simulation definition". Then run "Run simulations with range of total times/Generate data". <br>
ii. Run "Run simulations with range of total times/Import saved data". <br>
In either case, then run "Run simulations with range of total times/Total simulation times".
Then, all other cells from "Extract data (multiple samples)" can be run in any order.


Outputs are mostly printed in the notebook.
Commands to export simulation run data are included but commented out by default.
Most cells should take no more than a few seconds to run, with some of the heavier analysis taking somewhat longer.
The cell "Run simulations with range of total times/Generate data" may take around an hour to run, depending on the size of the largest network generated.

# LICENSE

This repository is licensed under the terms of the Creative Commons BY-NC 4.0 License.
