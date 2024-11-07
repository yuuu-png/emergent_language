#!/bin/bash

ENV_FILE=/work/bin/.generate_classify.env
while read line; do
    python3 /work/src/wandb/generate/classify.py --project=local_dev_run --checkpoint="${run}" --fast_dev_run
done < ${ENV_FILE}