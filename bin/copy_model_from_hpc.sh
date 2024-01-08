#!/usr/bin/env bash

rsync -e 'ssh -q' \
  -av --progress -h \
  dtu-hpc-transfer:~/dev/dtu-speechsep/data/models/$1/version_$2 data/models/$1
