#!/bin/bash
echo SRUN: python3 make_captions.py "$@"

srun --container-image=/netscratch/nauen/images/image_captioning_v2.sqsh \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds-sds:/ds-sds:ro,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
  --container-workdir="$(pwd)" \
  --time=1-0 \
  --mem=32G \
  --gpus=1 \
  --partition=A100-80GB,H100,RTXA6000,A100-40GB \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --export="HF_HOME=/netscratch/nauen/HF_HOME/" \
  --job-name="ImageCaptioning" \
  python3 make_captions.py "$@"
