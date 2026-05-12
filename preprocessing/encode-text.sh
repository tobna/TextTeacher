#!/bin/bash

#SBATCH --output=/netscratch/nauen/slurm/%x-%j-%N.out
#SBATCH --partition=RTXA6000,A100-PCI,H100-PCI
#SBATCH --job-name="create text encoding"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=1-0
#SBATCH --export=ALL,TQDM_DISABLE=1

srun -K \
  --container-image=/netscratch/nauen/images/dino_v1.1.sqsh \
  --container-workdir="$(pwd)" \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/fscratch/$USER:/fscratch/$USER,/ds-sds:/ds-sds:ro,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
  python3 encode_text.py "$@"
