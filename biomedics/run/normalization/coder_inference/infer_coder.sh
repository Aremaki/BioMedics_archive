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
source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh # appel de ce script
cd "/export/home/cse200055/Etienne/BioMedics/scripts/normalization"
source "/export/home/cse200055/Etienne/BioMedics/.venv/bin/activate"
conda deactivate

start_time="$(date -u +%s)"
input_dir="/export/home/cse200055/brat_data/Etienne/ai_triomph/EDS-biomedic/CONS"

echo -----------------
echo NORMALIZE BIO LABELS TOTAL APHP DOCS
echo -----------------
model_name="GanjinZero/coder_all"
output_dir="/export/home/cse200055/Etienne/ai_triomph/data/extracted/bio_cr_cons_test.json"
config_path="/export/home/cse200055/Etienne/BioMedics/scripts/normalization/coder_inference/bio_config.yml"

python coder_inference/run.py $model_name $input_dir $output_dir $config_path

echo -----------------
echo NORMALIZE MED LABELS TOTAL APHP DOCS

echo -----------------
drug_dict_path="/export/home/cse200055/Etienne/BioMedics/data/drug_knowledge/final_dict.pkl"
output_dir="/export/home/cse200055/Etienne/ai_triomph/data/extracted/drugs_cr_cons_test.json"
python fuzzy/run.py $drug_dict_path $input_dir $output_dir Chemical_and_drugs True jaro_winkler 0.8

end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "Total of $elapsed seconds for entities normalization"

echo --EXTRACTION_FINISHED---
echo ---------------
