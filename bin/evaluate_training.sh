#!/usr/bin/env bash

if [ -d "venv" ]
then
  source venv/bin/activate
fi

PYTHONPATH=. python speechsep/evaluation/training.py data/models/librimix/ $@
