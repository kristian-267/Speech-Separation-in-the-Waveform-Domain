#!/usr/bin/env bash

# Old
#checkpoint_path=data/models/sinusoid/version_19/checkpoints/epoch=119-step=30720.ckpt
#skip_upsampling=""

# with resampling
#checkpoint_path=data/models/sinusoid/version_0/checkpoints/epoch=149-step=4800.ckpt
#skip_upsampling=""

# without resampling
checkpoint_path=data/models/sinusoid/version_1/checkpoints/epoch=149-step=4800.ckpt
skip_upsampling="--skip-upsampling"

bin/predict.sh \
  --dataset sinusoid \
  --valid-length extend \
  ${skip_upsampling} \
  --checkpoint-path ${checkpoint_path} \
  --item 23
