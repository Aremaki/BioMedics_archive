from measurements_patterns import *
import pandas as pd


######################
# ## GENERAL CONFIG ###
# #####################
data_path = "/export/home/cse200093/Jacques_Bio/data_bio/normalisation_umls/snomed"
umls_path = "/export/home/cse200093/Jacques_Bio/data_bio/normalisation_umls_synonyms/bio_str_SNOMEDCT_US.json"
labels_column_name = "CUI"
# Name of the column which contains the CUIs
synonyms_column_name = "STR"
# Name of the column which contains the synonyms
temp_path = "/export/home/cse200093/Jacques_Bio/super_pipe/py_files/temp"
# temp_path is used to store files between each layer of the pipe.
# All files inside It will be deleted at the end of the treatment.
res_path = "/export/home/cse200093/Jacques_Bio/data_bio/super_pipe_res/res.json"
do_ner = False
do_measurements_pipe = False
do_coder = True


########################
# ## EDS MEDIC CONFIG ###
# #######################
eds_medic_model_path = "/export/home/cse200093/Pierre_Medic/NEURAL_BASED_NER/eds-medic/training/model-best"
eds_medic_gpu_id = 0
eds_medic_silent = False
eds_medic_output_path = temp_path


################################
# ## MEASUREMENTS PIPE CONFIG ###
# ###############################
measurements_pipe_regex_convert_spans = regex_convert_spans
measurements_pipe_label_key = label_key
measurements_pipe_labels_to_remove = labels_to_remove
measurements_pipe_labels_linkable_to_measurement = labels_linkable_to_measurement
measurements_pipe_config_normalizer_from_label_key = config_normalizer_from_label_key
measurements_pipe_config_measurements_from_label_key = config_measurements_from_label_key
measurements_pipe_config_normalizer_from_tables = config_normalizer_from_tables
measurements_pipe_config_measurements_from_tables = config_measurements_from_tables
measurements_pipe_brat_dir = temp_path
measurements_only_tables = False


####################
# ## CODER CONFIG ###
# ###################
df = None
# Pandas DataFrame which contains at least one column of
# terms or lists of terms which should be normalized (to the same CUI if It is a list)
# Useful only when we run CODER alone, else CODER will take the DataFrame
# from measurements pipe automatically
column_name_to_normalize = "terms_linked_to_measurement"
# Name of the preceding column of interest. Default should be
# "terms_linked_to_measurement" to make the entire pipe work
coder_model_name_or_path = "/export/home/cse200093/Jacques_Bio/data_bio/coder_output/model_150000.pth"
coder_tokenizer_name_or_path = "/export/home/cse200093/word-embedding/finetuning-camembert-2021-07-29"
coder_device = "cuda:0"
coder_save_umls_embeddings_dir = "/export/home/cse200093/Jacques_Bio/data_bio/normalisation_embeddings/umls_embeddings_normalized_snomed_coder_eds_2_cased.pt"
# set to False if you don't want to save
coder_save_umls_des_dir = "/export/home/cse200093/Jacques_Bio/data_bio/normalisation_embeddings/umls_des_normalized_snomed_coder_eds_2_cased.pkl"
# set to False if you don't want to save
coder_save_umls_labels_dir = "/export/home/cse200093/Jacques_Bio/data_bio/normalisation_embeddings/umls_labels_normalized_snomed_coder_eds_2_cased.pkl"
# set to False if you don't want to save
coder_save_data_embeddings_dir = "/export/home/cse200093/Jacques_Bio/data_bio/normalisation_embeddings/data_embeddings_normalized_snomed_coder_eds_2_cased.pt"
# set to False if you don't want to save
coder_normalize=True
coder_summary_method="CLS"
coder_tqdm_bar=True
coder_cased = True
coder_batch_size = 128
coder_stopwords = [
    "for",
    "assay",
    "by",
    "tests",
    "minute",
    "exam",
    "with",
    "human",
    "moyenne",
    "in",
    "to",
    "from",
    "analyse",
    "test",
    "level",
    "fluid",
    "laboratory",
    "determination",
    "examination",
    "releasing",
    "quantitative",
    "screening",
    "and",
    "exploration",
    "factor",
    "method",
    "analysis",
    "laboratoire",
    "specimen",
    "or",
    "typing",
    "of",
    "concentration",
    "measurement",
    "detection",
    "procedure",
    "identification",
    "numeration",
    "hour",
    "retired",
    "technique",
    "count",
]
coder_remove_stopwords_terms = False
coder_remove_special_characters_terms = False
coder_remove_stopwords_umls = True
coder_remove_special_characters_umls = True
