#!/bin/sh
### General options
### queue
#BSUB -q gpua100
### select GPU with enough memory
#BSUB -R "select[gpu32gb || gpu40gb || gpu80gb]"
### job name
#BSUB -J train_speechsep_librimix
### number of cores
#BSUB -n 4
### all cores on same host
#BSUB -R "span[hosts=1]"
### 2 GPUs in exclusive mode
#BSUB -gpu "num=2:mode=exclusive_process:mps=yes"
### walltime limit
#BSUB -W 24:00
### memory
#BSUB -R "rusage[mem=20GB]"
### notify upon start and completion
#BSUB -B -N
### output and error file
#BSUB -o data/lsf_logs/train_speechsep_librimix/%J.out
#BSUB -e data/lsf_logs/train_speechsep_librimix/%J.err

mkdir -p data/lsf_logs/train_speechsep_librimix

module load python3/3.9.14
module load cuda/11.7.1

nvidia-smi

bin/librimix/train.sh
