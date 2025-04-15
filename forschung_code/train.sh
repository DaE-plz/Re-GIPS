#!/bin/bash -l

#SBATCH --nodes=1               
#SBATCH --ntasks=1               

#SBATCH --time=23:00:00           
#SBATCH --job-name=xxdlgips      
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH -o /home/woody/iwi5/iwi5191h/forschung/dl_gips/model-%j.out
#SBATCH -e /home/woody/iwi5/iwi5191h/forschung/dl_gips/model-%j.err


#SBATCH --export=NONE

unset SLURM_EXPORT_ENV


module load python
conda activate forschung


mkdir /scratch/$SLURM_JOB_ID
mkdir /scratch/$SLURM_JOB_ID/ap
mkdir /scratch/$SLURM_JOB_ID/lt

tar -xvf /home/woody/iwi5/iwi5191h/projection_ap.tar.gz -C /scratch/$SLURM_JOB_ID/ap/
tar -xvf /home/woody/iwi5/iwi5191h/projection_lt.tar.gz -C /scratch/$SLURM_JOB_ID/lt/



#export CUDA_VISIBLE_DEVICES=1

python train_dl_gips.py

rm -r /scratch/$SLURM_JOB_ID