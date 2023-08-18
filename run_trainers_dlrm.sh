#!/bin/bash
batch_size=$1
master_ip=$2
workers=$3
log=$4
source /home/ubuntu/ptorch/bin/activate
torchrun --nnodes $workers --nproc_per_node 1 --rdzv_backend c10d --rdzv_endpoint $master_ip --rdzv_id 54321 --role trainer examples/trainer_main.py --batch-size $batch_size --world-size-trainers $workers --logging-prefix $log --s3 2>&1 | tee out
