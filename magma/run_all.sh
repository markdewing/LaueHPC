#!/bin/bash

EXE_PATH=./testing

for EXE in testing_dgeqrf_batched testing_dgesv_rbt_batched testing_dgetrf_batched testing_dgetrf_dgetrs_batched testing_dgetrs_batched
do
  NUM_ITER=5
  NUM_ROWS=798
  NUM_COLS=256

  # -- Header of summary file
  ${EXE_PATH}/${EXE} -n ${NUM_ROWS},${NUM_COLS} --batch 1 --niter 1 > out.txt
  head -9 out.txt > summary.txt


  BATCH_SIZE=1
  while [ ${BATCH_SIZE} -le 8192 ] 
  do
    COMMAND="${EXE_PATH}/${EXE} -n ${NUM_ROWS},${NUM_COLS} --batch ${BATCH_SIZE} --niter ${NUM_ITER}"
    echo "Executing COMMAND= ${COMMAND}"
    ${COMMAND} | tee out.txt
    tail -2 out.txt | head -1 >> summary.txt
    BATCH_SIZE=$(( BATCH_SIZE * 2 ))
  done

  rm -f out.txt
 
  mv summary.txt summary_${EXE}_niter_${NUM_ITER}.txt 
done
