Unravel
=======

Tools for unravelling wicked problems with graphical causal models.

repositry is at https://github.com/FjalarDeHaan/unravel

git clone https://github.com/FjalarDeHaan/unravel.git

Anaconda <br />
pip install cdt <br />
pip install pyvis <br />
pip install torch <br />
pip install pyCausalFS <br />
pip install pyreadstat <br />
pip install pyreadr <br />

R <br />
install.packages("pcalg") <br />
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager") BiocManager::install("RBGL") <br />
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager") BiocManager::install("Rgraphviz") <br />
Install.packages('devtools') <br />
Library(devtools) <br />
install_url('https://cran.r-project.org/src/contrib/Archive/CAM/CAM_1.0.tar.gz') <br />
install_url('https://cran.r-project.org/src/contrib/Archive/ccdrAlgorithm/ccdrAlgorithm_0.0.6.tar.gz') <br />
install_url('https://cran.r-project.org/src/contrib/Archive/sparsebn/sparsebn_0.1.2.tar.gz') <br />
install.packages('MASS') <br />
install.packages('momentchi2') <br />
install_github("Diviyan-Kalainathan/RCIT") <br />
install.packages('kpcalg') <br />

Causal.py <br />
Add below line after import CDT, with your R path <br />
cdt.SETTINGS.rpath = 'C:/Program Files/R/R-4.3.2/bin/Rscript' <br />

Download TEP data from https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset/data <br />
Place under unravel/TEP2017R <br />
(Maybe use lfs later) <br />

For Analyse_data <br />
pip install pgmpy <br />
pip install pygraphviz <br />
