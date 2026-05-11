#!/bin/bash

#SBATCH --output=/netscratch/nauen/slurm/%x-2024-11-15-09-59-29-%j-%N.out
#SBATCH --partition=batch,RTXA6000,RTX3090
#SBATCH --job-name="python"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=0-2
#SBATCH --export=NLTK_DATA=/netscratch/nauen/NLTK_DATA/,TQDM_DISABLE=1

srun -K \
  --container-image=/netscratch/nauen/images/custom_ViT_v14.sqsh \
  --container-workdir="`pwd`" \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/fscratch/$USER:/fscratch/$USER,/ds-sds:/ds-sds:ro,/ds:/ds:ro,"`pwd`":"`pwd`" \
  python3 "$@"
