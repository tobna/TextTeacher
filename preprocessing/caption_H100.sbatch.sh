#!/bin/bash

#SBATCH --array=0-3
#SBATCH --time=0-12
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=H100,H100-RP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name="Caption dataset: CoCa"
#SBATCH --output=/netscratch/nauen/slurm/%x-%j-%N-%a.out

srun --container-image=/netscratch/nauen/images/image_captioning_coca.sqsh \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds-sds:/ds-sds:ro,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
  --container-workdir="$(pwd)" \
  python3 make_captions.py -id $SLURM_ARRAY_TASK_ID -w 100 -m "CoCa" -b 32 -c "$@"
