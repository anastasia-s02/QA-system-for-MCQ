#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu
#SBATCH --job-name=GenMC

module purge

singularity exec --nv \
	    --overlay /scratch/tw2250/gen_mc/leo.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "python3 GenMC-main/run_genmc.py --model_path t5-base --gpu 0 --choice_num 4 --lr 5e-5 --data_path_train train_data_QA.jsonl --data_path_dev test_data_QA.jsonl  --data_path_test test_data_QA.jsonl  --train_batch_size 8  --gradient_accumulation_steps 1  --max_len 300  --max_len_gen 32 --alpha 1 --beta 0.5 --seed 1  --name_save_prix Gen_mc_test"