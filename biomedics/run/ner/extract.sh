#!/bin/bash
#SBATCH --job-name=coder_inference
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000
#SBATCH --partition gpuV100
#SBATCH --output=/export/home/cse200055/Etienne/ai_triomph/log/slurm-%x-%j-stdout.log
#SBATCH --error=/export/home/cse200055/Etienne/ai_triomph/log/slurm-%x-%j-stderr.log
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh  --container-mounts=/export/home/$USER:/export/home/$USER,/data/scratch/$USER:/data/scratch/$USER --container-mount-home --container-writable

nvidia-smi

source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh # appel de ce script
cd "/export/home/cse200055/Etienne/BioMedics"
source "env/bin/activate"
conda deactivate
pip install cupy
echo "Python used: $(which python)"

start_time="$(date -u +%s)"
input_dir="/export/home/cse200055/brat_data/Etienne/ai_triomph/EDS-biomedic/CONS"
output_path="/export/home/cse200055/Etienne/ai_triomph/data/extracted/CONS_ents"

python biomedics/ner/extract.py $input_dir --output-path $output_path

end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "Total of $elapsed seconds for entities extraction"

echo --EXTRACTION_FINISHED---
echo ---------------
