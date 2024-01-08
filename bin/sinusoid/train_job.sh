#!/bin/sh
### General options
### queue
#BSUB -q gpua40
### job name
#BSUB -J train_speechsep_sinusoid
### number of cores
#BSUB -n 4
### all cores on same host
#BSUB -R "span[hosts=1]"
### 2 GPUs in exclusive mode
#BSUB -gpu "num=2:mode=exclusive_process:mps=yes"
### walltime limit
#BSUB -W 10:00
### memory
#BSUB -R "rusage[mem=10GB]"
### notify upon completion
#BSUB -N
### output and error file
#BSUB -o data/lsf_logs/train_speechsep_sinusoid/%J.out
#BSUB -e data/lsf_logs/train_speechsep_sinusoid/%J.err

mkdir -p data/lsf_logs/train_speechsep_sinusoid

module load python3/3.9.14
module load cuda/11.7.1

nvidia-smi

bin/sinusoid/train.sh
