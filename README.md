# Thesis code: implementing DMERA on classical simulation and QC

This is the final version of the code that was used for the MSc thesis of Bjarne Bouwer, supervised by Michael Walter and Jonas Helsen. 
The code can be used to reproduce the data and figures that are in the thesis.

### Overview

The code included in this repo can perform:
- Classical simulation of DMERA (`classical_simulation.ipynb`)
- Eigenvalue extraction (Algorithm 2 from thesis) from measurement data (`extract_scalingdims.py`)
- Local simulation of DMERA quantum circuit (`braket/local_dmera.py`)
- Sending quantum tasks to QC via AWS (`braket/remote_dmera.py`)  
**NB**: need own Amazon account/credentials to send task to AWS

`native_gate_decomp.py` stores the gates that are used in classical simulation, including gates for the two noise models. 

The quantum circuit that is used to implement DMERA on QC can be found in `quantum_circuits.py`. There is an experimental option to do three layers, 
however this uses a cooling algorithm which is not optimal and will yield an energy value for 3 layers that is larger than the value at 2 layers. 

The file `make_figures.py` processes the data as obtained from the QC and produces a figure of the energies from QC (Figure 4.5). It also
produces the figures 4.3a and 4.3b.  

The results from the QC tasks that was used in the thesis (and previous experiments) are stored in the folder `/braket/results`, sorted by date. 
This folder includes a brief description on the experiments that were performed.

### Eigenvalue extraction algorithm

Guide for using the eigenvalue extraction algorithm (in `extract_scalingdims.py`):
1. Obtain decay curve data from `classical_simulation.ipynb` by running the function `create_all_decay_files()` for the desired decomposition
(or use data from the folder `/observable_data`)
2. From the Python file `extract_scalingdims.py`, run the function `comp_evals_from_all_files()` to use the algorithm as described in Section 5.2.3 with 
combining of the curves as described in Section 5.2.4. 


