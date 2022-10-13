#
# Definition of MACROS

CXX = CC
CXXFLAGS = -g -fopenmp -O3

CPP = cpp -P -traditional
CPPFLAGS =

LD = $(CXX)
LDFLAGS = $(CXXFLAGS)
LIB = 

ARCH ?= polaris
include ./arch/$(ARCH)

BINROOT=./
SHELL=/bin/sh

#
# -- target : 	Dependencies
# --		Rule to create target

EX = testing_dgeqrf_batched testing_dgesv_rbt_batched testing_dgetrf_batched testing_dgetrf_dgetrs_batched testing_dgetrs_batched

all : $(EX)

# -- 

testing_dgeqrf_batched: testing_dgeqrf_batched.o
	$(LD) -o $@ $(LDFLAGS) testing_dgeqrf_batched.o $(LIB)

testing_dgeqrf_batched.o: testing_dgeqrf_batched.cpp
	$(CXX) $(CXXFLAGS) -c $<

# -- 

testing_dgesv_rbt_batched: testing_dgesv_rbt_batched.o
	$(LD) -o $@ $(LDFLAGS) testing_dgesv_rbt_batched.o $(LIB)

testing_dgesv_rbt_batched.o: testing_dgesv_rbt_batched.cpp
	$(CXX) $(CXXFLAGS) -c $<

# --

testing_dgetrf_batched: testing_dgetrf_batched.o
	$(LD) -o $@ $(LDFLAGS) testing_dgetrf_batched.o $(LIB)

testing_dgetrf_batched.o: testing_dgetrf_batched.cpp
	$(CXX) $(CXXFLAGS) -c $<

# --

testing_dgetrf_dgetrs_batched: testing_dgetrf_dgetrs_batched.o
	$(LD) -o $@ $(LDFLAGS) testing_dgetrf_dgetrs_batched.o $(LIB)

testing_dgetrf_dgetrs_batched.o: testing_dgetrf_dgetrs_batched.cpp
	$(CXX) $(CXXFLAGS) -c $<

# --

testing_dgetrs_batched: testing_dgetrs_batched.o
	$(LD) -o $@ $(LDFLAGS) testing_dgetrs_batched.o $(LIB)

testing_dgetrs_batched.o: testing_dgetrs_batched.cpp
	$(CXX) $(CXXFLAGS) -c $<

#
# -- Remove *.o and *~ from the directory
clean:
	rm -f *.o *~
#
# -- Remove *.o, *~, and executable from the directory
realclean:
	rm -f *.o *~ $(EX)
