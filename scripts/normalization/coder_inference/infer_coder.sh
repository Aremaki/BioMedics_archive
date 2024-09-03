#!/bin/bash
#SBATCH --job-name=coder_inference
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000
#SBATCH --partition gpuV100
#SBATCH --output=log_coder_infer/slurm-%j-stdout.log
#SBATCH --error=log_coder_infer/slurm-%j-stderr.log
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh  --container-mounts=/export/home/$USER:/export/home/$USER,/data/scratch/$USER:/data/scratch/$USER --container-mount-home --container-writable
source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh # appel de ce script
cd "/export/home/cse200093/Adam/biomedics/scripts/normalization"
source "/export/home/cse200093/Adam/biomedics/.venv/bin/activate"
conda deactivate

echo -----------------
echo NORMALIZE BIO LABELS TOTAL APHP DOCS
echo -----------------

python coder_inference/run.py /data/scratch/cse200093/word-embedding/coder_all ../../data/annotated_CRH/post_processed/expe_complete_pipe/pred/NER ../../data/annotated_CRH/post_processed/expe_complete_pipe/pred/NER_Norm/pred_bio_coder_all.json BIO

echo -----------------
echo NORMALIZE MED LABELS TOTAL APHP DOCS
echo -----------------

python fuzzy/run.py /data/scratch/cse200093/word-embedding/coder_all ../../data/annotated_CRH/post_processed/expe_complete_pipe/pred/NER ../../data/annotated_CRH/post_processed/expe_complete_pipe/pred/NER_Norm/pred_bio_coder_all.json BIO
python ./Normalisation/fuzzy/main.py ../../data/drug_knowledge/final_dict.pkl ../../data/annotated_CRH/post_processed/expe_complete_pipe/pred/NER  ../../data/annotated_CRH/post_processed/expe_complete_pipe/pred/NER_Norm/pred_med_fuzzy_jw.json Chemical_and_drugs True jaro_winkler 0.8

start_time="$(date -u +%s)"

echo -----------------
echo NORMALIZE BIO AND MED
echo -----------------

for disease in "lupus_erythemateux_dissemine" "maladie_de_takayasu" "sclerodermie_systemique" "syndrome_des_anti-phospholipides"
do

    echo -----------------
    echo PROCESS $disease
    echo -----------------

    echo -----------------
    echo NORMALIZE BIO LABELS
    echo -----------------

    python coder_inference/run.py /data/scratch/cse200093/word-embedding/coder_all ./data/bio_results_v3/$disease/pred_with_measurement.pkl ./data/bio_results_v3/$disease/pred_bio_coder_all.json BIO

    echo -----------------
    echo NORMALIZE MED LABELS
    echo -----------------

    python ./Normalisation/fuzzy/main.py ./data/drugs_resources/final_dict.pkl ./data/CRH/pred_v3/$disease ./data/bio_results_v3/$disease/pred_med_fuzzy_jaro_winkler.json Chemical_and_drugs True jaro_winkler 0.8
    end_time="$(date -u +%s)"
    elapsed="$(($end_time-$start_time))"
    echo "Total of $elapsed seconds elapsed for $disease"

done

end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "Total of $elapsed seconds elapsed for process"

echo --EXTRACTION_FINISHED---

echo ---------------
