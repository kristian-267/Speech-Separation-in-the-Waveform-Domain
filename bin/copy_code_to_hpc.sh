#!/usr/bin/env bash

rsync -e 'ssh -q' \
  -av \
  --exclude __pycache__ --exclude *.pyc \
  speechsep bin requirements.txt pyproject.toml \
  dtu-hpc-transfer:~/dev/dtu-speechsep
