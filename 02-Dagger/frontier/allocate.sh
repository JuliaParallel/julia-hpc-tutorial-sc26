#!/bin/bash
# STUDENTS: grab your compute allocation at the START of the tutorial, so no
# exercise time is lost waiting in the queue.
#   bash frontier/allocate.sh
# TODO(confirm): account, reservation, and per-user shape (assumed: 1 GPU + a few cores).
salloc -A "${SC26_ACCOUNT:-TRN047}" \
       ${SC26_RESERVATION:+--reservation=$SC26_RESERVATION} \
       -N 1 -n 2 -c 3 --gpus=1 -t 02:00:00
