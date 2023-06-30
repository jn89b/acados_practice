# Practice with Acados 

# Installation procedure 
- Refer to link https://docs.acados.org/installation/index.html
- Follow the Linux Tutorials on how to build
- During the cmake portion compile with the following commands instead

```
cd <acados_dir>
rm build/* -rf
cd build
cmake .. -DACADOS_WITH_QPOASES=ON -DACADOS_EXAMPLES=ON -DHPIPM_TARGET=GENERIC -DBLASFEO_TARGET=GENERIC
make -j4
make install -j4
# run a C example, e.g.:
./examples/c/sim_wt_model_nx6
```
-l During the installation BLASFEO_TARGET you can check that by entering the following command in your terminal:
  - lscpu


## Dependency issues
- Need to install cython or upgrade it with the following:
  - pip3 install --upgrade cython
- Need to install future-fstrings
