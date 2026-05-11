#!/bin/bash

#SBATCH --output=/netscratch/nauen/slurm/%x-2025-08-28-15-21-31-%j-%N.out
#SBATCH --partition=A100-PCI,RTX3090,RTXA6000,RTXA6000-SDS,H100-PCI,L40S
#SBATCH --job-name="CKA analysis"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-2
#SBATCH --export=ALL,NLTK_DATA=/netscratch/$USER/NLTK_DATA/,TQDM_DISABLE=1,HF_HOME=/fscratch/nauen/HF_HOME/

srun -K \
  --container-image=/netscratch/nauen/images/custom_ViT_v2.3.sqsh \
  --container-workdir="$(pwd)" \
  --container-mounts=/netscratch/nauen:/netscratch/nauen,/fscratch/nauen:/fscratch/nauen,/ds-sds:/ds-sds:ro,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
  python3 cka.py -m1 "$@"
