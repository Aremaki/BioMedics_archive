## How to train the model ##

conda activate pierrenv

go on this path: Pierre_Medic/NEURAL_BASED_NER

## Project.yml ##
Pierre_Medic/NEURAL_BASED_NER/eds-medic/project.yml

Modifier les path du dataset: 
  train: "../data/attr/train"
  dev: "../data/attr/val"
  test: "../data/attr/test"
  
  
## Config.cfg ##
Pierre_Medic/NEURAL_BASED_NER/eds-medic/configs/config.cfg

Modifier ent_label: 
ent_labels =["DISO","BIO_comp","Chemical_and_drugs","dosage","BIO","strength","form","SECTION_antecedent","route","SECTION_motif","SECTION_histoire","BIO_milieu","SECTION_examen_clinique","SECTION_examen_complementaire","SECTION_mode_de_vie","SECTION_traitement_entree","SECTION_antecedent_familiaux","SECTION_traitement_sortie","SECTION_traitement","SECTION_evolution","SECTION_autre","SECTION_conclusion"]

Modifier les attributs et le linkage attrributs--entités

qualifiers = ["_.Action","_.Certainty","_.Temporality","_.Negation","_.Family"]
label_constraints = {"_.Action":["Chemical_and_drugs"],"_.Certainty":["Chemical_and_drugs"],"_.Temporality":["Chemical_and_drugs"],"_.Negation":["Chemical_and_drugs"],"_.Family":["Chemical_and_drugs"]}


## 
cd Pierre_Medic/NEURAL_BASED_NER
sbatch train.sh 
-- 
Pierre_Medic/NEURAL_BASED_NER/eds-medic/scripts/save_to_brat.py
a modifier pour mettre le bon path 

--
sbatch test.sh

sbatch save.sh 

## Console display ##

tail -f -n 100000000000000 logs/slurm-10953-stdout.log 


## commande gpu ##
scancel : pour cancel un job
