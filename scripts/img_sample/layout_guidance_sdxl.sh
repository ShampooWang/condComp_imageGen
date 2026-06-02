#!/bin/bash
#SBATCH --job-name=sdxl_sample-lg_cls2_seed322 ## job name
#SBATCH --nodes=1                ## 索取 2 節點
#SBATCH --ntasks-per-node=1      ## 每個節點運行 8 srun tasks
#SBATCH --cpus-per-task=4        ## 每個 srun task 索取 4 CPUs
#SBATCH --gres=gpu:4            ## 每個節點索取 8 GPUs
#SBATCH --account="MST114289"   ## PROJECT_ID 請填入計畫ID(ex: MST108XXX)，扣款也會根據此計畫ID
#SBATCH --partition=gp2d        # gtest 為測試用 queue，後續測試完可改 gp1d(最長跑1天)、gp2d(最長跑2天)、gp4d(最長跑4天)
#SBATCH --output=logs/test/lg_cls2_seed322_3.log

class_count=2
seed=322
accelerate launch --multi_gpu --num_processes 4 sdxl_sample.py \
    --total_images 8 \
    --out_dir samples/test/lg/cls${class_count}/seed${seed}_3 \
    --seed ${seed} \
    --class_count ${class_count} \
    --layout_guidance_sdxl 
    
    