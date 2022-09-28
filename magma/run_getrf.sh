#!/bin/bash

# Batched LU Factorization
EXE=~/soft/thetagpu/magma/magma-2.6.2/build/testing/testing_dgetrf_batched

NUM_ROWS=798
NUM_COLS=256

# -- Header of summary file
${EXE} -n ${NUM_ROWS},${NUM_COLS} --batch 1 --niter 1 > out.txt
head -9 out.txt > summary.txt


BATCH_SIZE=1
while [ ${BATCH_SIZE} -le 4096 ] 
do
  COMMAND="${EXE} -n ${NUM_ROWS},${NUM_COLS} --batch ${BATCH_SIZE} --niter 3"
  echo "Executing COMMAND= ${COMMAND}"
  ${COMMAND} | tee out.txt
  tail -2 out.txt | head -1 >> summary.txt
  BATCH_SIZE=$(( BATCH_SIZE * 2 ))
done


rm -f out.txt
