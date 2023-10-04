#!/bin/bash 
# SBATCH --job-name=main_with_parts.sh
# SBATCH -t 24:00:00 
# SBATCH --gres=gpu:t4:1
# SBATCH -N1-1 
# SBATCH -c2
# SBATCH --mem=40000 
# SBATCH -p gpuT4
# SBATCH -w bbs-edsg28-p012
# SBATCH --output=./log/%x-%j.out
# SBATCH --error=./log/%x-%j.err
# SBATCH --container-image /scratch/images/sparkhadoop.sqsh  --container-mounts=/export/home/$USER:/export/home/$USER,/data/scratch/$USER:/data/scratch/$USER --container-mount-home --container-writable 

# Loading conda env
source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh

# Execute NER
# echo Starting NER
# conda activate pierrenv
# python /export/home/cse200093/Jacques_Bio/super_pipe/py_files/main_part_1.py
# conda deactivate
# echo NER finished

# Execute Measurements pipe
# echo Starting measurements pipe
# conda activate jacques-py37-spark
# python /export/home/cse200093/Jacques_Bio/super_pipe/py_files/main_part_2.py
# conda deactivate
# echo Measurements pipe finished

# Execute CODER
echo Starting CODER
conda activate pierrenv
python /export/home/cse200093/Jacques_Bio/super_pipe/py_files/main_part_3.py
conda deactivate
echo CODER finished
