#!/bin/bash

srun -K \
  --container-image=/netscratch/nauen/images/custom_ViT_v2.0.sqsh \
  --container-workdir="$(pwd)" \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/fscratch/$USER:/fscratch/$USER,/ds-sds:/ds-sds:ro,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
  --partition=RTXA6000,RTX3090,A100-40GB,A100-80GB,H100,A100-SDS,A100-RP,H100-RP,H200,H200-SDS \
  --job-name="python" \
  --nodes=1 \
  --gpus=0 \
  --ntasks=1 \
  --cpus-per-task=24 \
  --mem=64G \
  --time=1-0 \
  --export="NLTK_DATA=/netscratch/nauen/NLTK_DATA/" \
  python3 "$@"
