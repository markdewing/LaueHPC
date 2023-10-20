
# Assumes MKL has been loaded as a module

g++ \
-O3 \
-g \
-D EIGEN_DONT_PARALLELIZE \
-I /home/mdewing/.local/lib/python3.10/site-packages/pybind11/include/ \
-I /usr/include/python3.10/ \
-I /home/mdewing/software/linalg/eigen/eigen/ \
-shared \
-fPIC \
-o solver.so \
pyeigen1.cpp \
from_chatgpt2.cpp  \
-Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core  -lpthread -lm -ldl


#-L /usr/lib/python3.10/config-3.10-x86_64-linux-gnu/ \
#-l python3.10

