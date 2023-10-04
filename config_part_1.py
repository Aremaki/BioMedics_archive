import pandas as pd


######################
# ## GENERAL CONFIG ###
# #####################
data_path = "/export/home/cse200093/Jacques_Bio/data_bio/gold_std_NLP_diabeto/annotation"
umls_path = "/export/home/cse200093/Jacques_Bio/data_bio/normalisation_umls_synonyms/bio_str_SNOMEDCT_US.json"
labels_column_name = "CUI"
# Name of the column which contains the CUIs
synonyms_column_name = "STR"
# Name of the column which contains the synonyms
temp_path = "/export/home/cse200093/Jacques_Bio/data_bio/gold_std_NLP_diabeto/pred_ner"
# temp_path is used to store files between each layer of the pipe.
# All files inside It will be deleted at the end of the treatment.
res_path = "/export/home/cse200093/Jacques_Bio/data_bio/super_pipe_res/stats_by_section_on_cim10/gold_std_NLP_diabeto/res.json"


########################
# ## EDS MEDIC CONFIG ###
# #######################
eds_medic_model_path = "/export/home/cse200093/Pierre_Medic/NEURAL_BASED_NER/inference_model/model-best"
eds_medic_gpu_id = 0
eds_medic_silent = False
eds_medic_output_path = temp_path
