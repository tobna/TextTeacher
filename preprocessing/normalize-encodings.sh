#!/bin/bash

srun -K --partition=RTXA6000,V100-32GB,A100-PCI,H100-PCI --job-name="normalize encodings" --nodes=1 --gpus=0 --ntasks=1 --cpus-per-task=24 --mem=256G --time=1-0 --container-image=/netscratch/nauen/images/dino_v1.1.sqsh --export=NLTK_DATA=/netscratch/nauen/NLTK_DATA/,HF_HOME=/fscratch/nauen/HF_HOME/ --container-workdir="$(pwd)" \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/fscratch/$USER:/fscratch/$USER,/ds-sds:/ds-sds:ro,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
  python3 make_encoding_normalizer.py -f "$@"
