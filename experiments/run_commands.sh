
## Fine Tuning (on final.tar)
./main.py -task fine-tune -model /netscratch/nauen/EfficientCVBench/models/pretrain_ -lr 3e-4 -epochs 50 -dataset imagenet -experiment_name EfficientCVBench -run_name "-S @224" -container-image /netscratch/nauen/images/custom_ViT_v9.2.sqsh -no_comp -mem-per-gpu 110

## Efficiency Metrics (on top.tar)
./main.py -task eval-metrics -ntasks 1 -partition A100-RP A100-80GB -time 0-2 -batch_size 512 -container-image /netscratch/nauen/images/custom_ViT_v9.2.sqsh -num_workers 4 -dataset imagenet -no_comp -model /netscratch/nauen/EfficientCVBench/models/finetune_ -imsize 224

./main.py -task eval-metrics -ntasks 1 -partition RTX3090 -time 0-2 -batch_size 512 -container-image /netscratch/nauen/images/custom_ViT_v9.2.sqsh -num_workers 4 -dataset imagenet -no_comp -model /netscratch/nauen/EfficientCVBench/models/finetune_ -imsize 224 -new_log

## Down-Stream Datasets (on final.tar)
./main.py -task fine-tune -model /netscratch/nauen/EfficientCVBench/models/finetune_ -lr 3e-4 -epochs 2000 -dataset stanford-cars -experiment_name EfficientCVBench -run_name "-S Cars" -container-image /netscratch/nauen/images/custom_ViT_v9.2.sqsh -no_comp -ntasks 2 -batch_size 1024 -save_epochs 200 -mem-per-gpu 110 -time 1-0

./main.py -task fine-tune -model /netscratch/nauen/EfficientCVBench/models/finetune_ -lr 3e-4 -epochs 2000 -dataset flowers102 -experiment_name EfficientCVBench -run_name "-S Flowers" -container-image /netscratch/nauen/images/custom_ViT_v9.2.sqsh -no_comp -ntasks 1 -batch_size 256 -save_epochs 200 -mem-per-gpu 110 -time 1-0

./main.py -task fine-tune -model /netscratch/nauen/EfficientCVBench/models/finetune_ -lr 3e-4 -epochs 50 -dataset places365 -experiment_name EfficientCVBench -run_name "-S Places" -container-image /netscratch/nauen/images/custom_ViT_v9.2.sqsh -no_comp -ntasks 4 -batch_size 2048 -mem-per-gpu 110 -time 1-0 -dataset_root /ds/images/ -num_workers 20
