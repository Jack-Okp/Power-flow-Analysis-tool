# Power-flow-Analysis-tool
This script runs data to solve powerflow analysis using Newton Raphson method and Contigency analysis

BEFORE USAGE PLESE 
the code has some short comings, new updates will be committed soon

Numerical Stability:
Newton-Raphson method might not always converge
No handling for singular matrices in _build_jacobian()
Hardcoded tolerance and max iterations may not work for all networks

Contingency Analysis:
Removes and re-adds edges without preserving original edge attributes completely
Assumes successful Newton-Raphson convergence for each contingency

Potential Bug in Indexing:
Indexing assumes specific bus types ('PV', 'PQ')
May fail if bus types are not exactly as expected
Uses index-based access which can be fragile
