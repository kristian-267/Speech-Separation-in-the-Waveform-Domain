#!/usr/bin/env bash

if [[ -z "${LSF_ENVDIR}" ]]; then
  batch_size=16
else
  # On HPC
  batch_size=64
  device="--gpu --devices 2"
fi

bin/train.sh \
  --dataset sinusoid \
  --sinusoid-n-examples 32768 \
  --valid-length extend \
  --batch-size ${batch_size} \
  --max-epochs 150 \
  --dropout-p 0.4 \
  --checkpoint-every-n-epochs 50 \
  --limit-val-batches 4 \
  --log-every-n-steps 10 \
  --context 5 \
  --skip-upsampling \
  ${device}
