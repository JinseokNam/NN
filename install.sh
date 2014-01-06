#!/bin/bash

#####################################
# Install CSparse for sparse matrix
#####################################

# Download CSparse
wget https://www.cise.ufl.edu/research/sparse/CSparse/CSparse.tar.gz
tar xvzf CSparse.tar.gz
cd CSparse
make library

###################
# Install OpenBLAS
###################

# remove binary if installed
sudo apt-get remove libopenblas-base

cd /tmp/
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make NO_AFFINITY=1 USE_OPENMP=1
sudo make install
