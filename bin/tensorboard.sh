#!/usr/bin/env bash

if [ -d "venv" ]; then
   source venv/bin/activate
fi

if [[ -z "${LSF_ENVDIR}" ]]; then
   PORT="default"
else
   PORT=45659  # random port to avoid conflict
fi

tensorboard --logdir data/models --port $PORT --reload_multifile True
