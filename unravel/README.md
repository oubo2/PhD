Unravel
=======

Tools for unravelling wicked problems with graphical causal models.

repositry is at https://github.com/FjalarDeHaan/unravel

git clone https://github.com/FjalarDeHaan/unravel.git

Anaconda
pip install cdt
pip install pyvis
pip install torch
pip install pyCausalFS
pip install pyreadstat
pip install pyreadr

R
Install.packages("pcalg")
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager") BiocManager::install("RBGL")
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager") BiocManager::install("Rgraphviz")
Install.packages('devtools')
Library(devtools)
install_url('https://cran.r-project.org/src/contrib/Archive/CAM/CAM_1.0.tar.gz')
install_url('https://cran.r-project.org/src/contrib/Archive/ccdrAlgorithm/ccdrAlgorithm_0.0.6.tar.gz')
install_url('https://cran.r-project.org/src/contrib/Archive/sparsebn/sparsebn_0.1.2.tar.gz')
install.packages('MASS')
install.packages('momentchi2')
install_github("Diviyan-Kalainathan/RCIT")
install.packages('kpcalg')

Causal.py
Add below line after import CDT, with your R path
cdt.SETTINGS.rpath = 'C:/Program Files/R/R-4.3.2/bin/Rscript'

Download TEP data from https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset/data
Place under unravel/TEP2017R
(Maybe use lfs later)
