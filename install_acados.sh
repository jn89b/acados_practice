#!/bin/sh

#clone acados
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init

#build and install acados
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
# add more optional arguments e.g. -DACADOS_WITH_OSQP=OFF/ON -DACADOS_INSTALL_DIR=<path_to_acados_installation_folder> above
make install -j4

#install acados 
SCRIPTDIR="$(dirname "$0")"
make shared_library
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SCRIPTDIR/lib
make examples_c
make run_examples_c

