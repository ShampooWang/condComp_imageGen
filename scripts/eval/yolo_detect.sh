#!/bin/bash
#SBATCH --job-name=yolo_detect-lg_new_tokmap_cls6 ## job name
#SBATCH --nodes=1                ## 索取 2 節點
#SBATCH --ntasks-per-node=1      ## 每個節點運行 8 srun tasks
#SBATCH --cpus-per-task=4        ## 每個 srun task 索取 4 CPUs
#SBATCH --gres=gpu:1            ## 每個節點索取 1 GPUs
#SBATCH --account="MST114289"   ## PROJECT_ID 請填入計畫ID(ex: MST108XXX)，扣款也會根據此計畫ID
#SBATCH --partition=gp1d        ## gtest 為測試用 queue，後續測試完可改 gp1d(最長跑1天)、gp2d(最長跑2天)、gp4d(最長跑4天)
#SBATCH --output=logs/eval/val/lg_new_tokmap_cls6.log

# export CUDA_VISIBLE_DEVICES=7 # A5000


python obj_eval/yolo_detect_objects.py \
    --img_dir samples/test/seq_inpaint_val/cls2/seed322_4 \