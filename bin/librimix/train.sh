#!/usr/bin/env bash

if [[ -z "${LSF_ENVDIR}" ]]; then
  batch_size=16
  librimix_folder=/mnt/d/dev/LibriMix/Libri2Mix/wav8k/min/metadata
else
  # On HPC
  batch_size=64
  device="--gpu --devices 2"
  librimix_folder=/work3/projects/02456/project04/librimix/Libri2Mix/wav8k/min/metadata

  if ! command -v nvidia-smi &> /dev/null
  then
      echo "Need to run on HPC GPU node."
      exit
  fi
fi

bin/train.sh \
  --dataset librimix \
  --librimix-train-metadata \
      ${librimix_folder}/mixture_train-360_mix_both.csv \
      ${librimix_folder}/mixture_train-100_mix_both.csv \
  --librimix-val-metadata ${librimix_folder}/mixture_dev_mix_both.csv \
  --librimix-test-metadata ${librimix_folder}/mixture_test_mix_both.csv \
  --skip-normalization \
  --skip-upsampling \
  --context 3 \
  --example-length 4 \
  --batch-size ${batch_size} \
  --dropout-p 0 \
  --max-epochs 360 \
  --log-every-n-steps 10 \
  --checkpoint-every-n-epochs 10 \
  ${device} \
  $@
