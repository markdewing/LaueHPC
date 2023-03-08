
# Using the 'magma' module
MAGMA_LIB = -L/soft/libraries/magma-2.7.0/lib

# System python on Polaris
#PYTHON_INC = -I/usr/include/python3.6m
# Install pybind with "pip install --user pybind11"
#PYBIND_INC = -I/home/mdewing/.local/lib/python3.6/site-packages/pybind11/include

# Using 'cray-python' module on Polaris
# Should set using appropriate env variables (CRAY_PYTHON_PREFIX?)
PYTHON_INC = -I/opt/cray/pe/python/3.9.12.1/include/python3.9/
PYBIND_INC = -I/opt/cray/pe/python/3.9.12.1/lib/python3.9/site-packages/pybind11/include
