# Data set

Three public dataset are used by this study:

BICCN fMOST data from Brain Image Library (BIL): https://doi.brainimagelibrary.org/doi/10.35077/g.73 

Janelia MouseLight (JML): http://mouselight.janelia.org/

Allen Brain Cell Types (ACT): https://celltypes.brain-map.org/


Please cite the data above accordingly when using them in your study!


User needs to download the data from source first.
All data should organized under data folder.
Please refer to the description below 

## BICCN fMOST data from Brain Image Library
Data can be downloaded from: https://doi.brainimagelibrary.org/doi/10.35077/g.73
Run python script 'src/download_BIL.py' to automatically download the whole set.
All neuron reconstructions will be downloaded and saved in folder: data/raw/seu_nature/swc.

## Janelia MouseLight
Janelia MouseLight website provides link to download neuron reconstructions in bulk.
The link can be found in the settings of this webpage: http://ml-neuronbrowser.janelia.org/ (click top right settings button of the webpage).
Here is the latest link: http://ml-neuronbrowser.janelia.org/download/2022-02-28-json.tar.gz
Extract the file under folder data/raw/janelia_mouselight.
Makesure there is json files in folder data/raw/janelia_mouselight/json30.

## Allen Brain Cell Types
Run python script 'src/download_ACT.py'. 
All neuron reconstructions will be downloaded and saved in folder: data/raw/allen_cell_type/swc

## Data preprocess
After data download and verification, run python script 'src/prepare_data.py'.
All processed neuron reconstructions will be saved in folder data/dendrite
Neuron infos will be saved in folder data/info

## Notes
The python scripts we provided to download data are not official codes.
They are tools to simply the process.
Please refer to the data link above for the terms of use of each dataset.
