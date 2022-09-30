#!/bin/bash
#SBATCH --job-name=plugen-ae
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --partition=dgxmatinf

cd $HOME/plugen-ae/StyleGAN
source activate /home/bedychaj/anaconda3/envs/StyleFlow
python -u train_ae.py