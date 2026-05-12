#!/bin/bash
#SBATCH --partition=RTXA6000,V100-32GB,A100-PCI,H100-PCI
#SBATCH --job-name="normalize encodings"
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=1-0
#SBATCH --output=/netscratch/nauen/slurm/%x-%j-%N.out

srun -K --container-image=/netscratch/nauen/images/dino_v1.1.sqsh --container-workdir="$(pwd)" \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/fscratch/$USER:/fscratch/$USER,/ds-sds:/ds-sds:ro,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
  --export=NLTK_DATA=/netscratch/nauen/NLTK_DATA/,TQDM_DISABLE=1,HF_HOME=/fscratch/nauen/HF_HOME/ \
  python3 make_encoding_normalizer.py -f "$@"
