# Welcome to mysite

## Quick Background
The 2x2 is a small scale prototype detector of the proposed ND-LAr for the Deep Underground Neutrino Experiment (DUNE). DUNE is a neutrino oscillation experiment that will detect neutrinos of a one particular flavor at the near detector in Illinois and another flavor at the flavor at the far detector in South Dakota. However, as the neutrinos travel through the rock towards the near or far detector, some will interact with the rock producing muons which we will see in our detectors. This website will host the code that select these muons.

The selection has 3 main parts which is the selection of muons, the selection performance (purity and efficiency), and the electron lifetime. The selection will produce a csv file containing the track information, a csv file containing the segment information, a png of the purity and efficiency, and a plot of the electron lifetime. <br>

![Data Flow Diagram](./DFD.png)


## 1. Muon Selection <br>
The muon selection takes in hdf5 files produced by [ndlar-flow](https://github.com/DUNE/ndlar_flow) and loop through the events to select muon tracks.
Each event will be clustered using a DBSCAN algorithm and some clusters will be merged depending on their
linearity with another cluster. These clusters are then tested for their straightness (using principal component analysis), if they are through-going, and if they are at least one hundred centimeters. If so then the cluster is considered a muon, and the muon track will be split into 2 centimeters segments.

## 2. Selection Performance
### 2.1 Purity <br>
After the muon selection is done for a file, the purity of the selection is calculated. The purity is the amount of positively charged muons and negatively charges muon out of all the tracks that were selected.

### 2.2 Efficiency <br>
After the muon selection is done for a file, the efficiency of the selection is calculated. The efficiency is the amount of positively charged muons and negatively charges muon selected out of all the true muon tracks.

## 3. Electron Lifetime <br>
The electron lifetime using the rock muon segments dN/dx (number of hits per unit length) and dQ/dx (charge loss per unit length) is calculated the lifetime. The convolution of the dN/dx and dQ/dx is split into twenty time bins, and the MPV is taken for each time bin. The MPVs for each time bins are fitted with the electron lifetime function to extract the lifetime.
