#!/usr/bin/env bash

if [[ -z "${LSF_ENVDIR}" ]]; then
  librimix_folder=/mnt/d/dev/LibriMix/Libri2Mix/wav8k/min/metadata
else
  # On HPC
  librimix_folder=/work3/projects/02456/project04/librimix/Libri2Mix/wav8k/min/metadata
fi

# B=64, p=0
#checkpoint_path=data/models/librimix/version_32/checkpoints/epoch=359-step=173520.ckpt
# B=8, p=0
#checkpoint_path=data/models/librimix/version_36/checkpoints/epoch=359-step=1579270.ckpt
# B=64, p=0.2
checkpoint_path=data/models/librimix/version_38/checkpoints/epoch=359-step=346680.ckpt

# Training dataset
# test_dataset=/mnt/d/dev/LibriMix/Libri2Mix/wav8k/min/metadata/mixture_train-100_mix_both.csv
# Actual test dataset
test_dataset=${librimix_folder}/mixture_dev_mix_both.csv

bin/predict.sh \
  --dataset librimix \
  --librimix-test-metadata ${test_dataset} \
  --checkpoint-path ${checkpoint_path} \
  --valid-length pad \
  --skip-normalization \
  --skip-upsampling \
  --context 3 \
  --example-length 4 \
  --item 10
