#!/bin/bash 
#SBATCH --job-name=hvps_cnn_predictions_2      # Job name

#SBATCH --partition=standard      # !! Partition to submit to
#SBATCH --qos=long
#SBATCH --account=dcmex     # our gws space
#SBATCH --output=/home/users/esree/slurm_yerm/output/hvps_cnn_predictions_2_%j.txt
#SBATCH --error=/home/users/esree/slurm_yerm/error/hvps_cnn_predictions_2_%j.txt
#SBATCH --time= 3-12:00:00          # Wall clock limit (d-HH:MM:SS) i.e. 3 days and 12 hours
#SBATCH --mem=40G  # Request 32 GB of memory


## executable 
##sleep 5m
# Activate your virtual environment - in Jasmin
source ~/miniforge3/bin/activate # conda miniforge3 base environment
source activate Jaffeux_venv_2 # specific environment for running the CNN

python3 /home/users/esree/CNN_basics/Jaffeux_24_models/both_multi_model_run.py # the script for running the CNN

conda deactivate
conda deactivate
