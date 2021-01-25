# Guide for installing and running code
## Installation
- install anaconda on your machine
- create the environment:  
<code> conda env create --name ada_FISTA -f env.yml </code>
- for the STORM example, download the data from http://bigwww.epfl.ch/smlm/challenge2016/datasets/MT4.N2.HD/Data/data.html. The desired file is `MT4.N2.HD-2D-Exp-as-stack` which should be unzipped into the `data` directory. This process should create the file `.../code/data/sequence-as-stack-MT4.N2.HD-2D-Exp.tif`.

## Running examples
- move to the `code` directory
- set the python path:   <code> conda activate ada_FISTA </code>
- open a jupyter notebook:  <code> jupyter notebook </code> 
- open one of the `.ipynb` scripts
- click the '`>>`' button to 'restart the kernel, then re-run the whole notebook'
- two folders will be created for videos and figure images
- once the reconstructions have been computed, they are stored in `data` and reused to make plots