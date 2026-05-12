#!/bin/bash

#SBATCH --time=1-0
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=B200
#SBATCH --reservation=getai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name="FID score from seed"
#SBATCH --output=/netscratch/nauen/slurm/%x-%j-%N-%a.out

srun --container-image=/netscratch/nauen/images/image_captioning_v2.1.sqsh \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds-sds:/ds-sds:ro,/ds:/ds:ro,/fscratch/$USER:/fscratch/$USER,"$(pwd)":"$(pwd)" \
  --container-workdir="$(pwd)" \
  python3 -u seed_fid.py "$@"
