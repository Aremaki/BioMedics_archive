from measurements_patterns import *
import pandas as pd


######################
# ## GENERAL CONFIG ###
# #####################
umls_path = "/export/home/cse200093/Jacques_Bio/data_bio/normalisation_umls_synonyms/bio_str_SNOMEDCT_US.json"
labels_column_name = "CUI"
# Name of the column which contains the CUIs
synonyms_column_name = "STR"
# Name of the column which contains the synonyms
temp_path = "/export/home/cse200093/Jacques_Bio/data_bio/super_pipe_get_stats_by_section_on_cim10/pred/maladie_de_takayasu_pred"
# temp_path is used to store files between each layer of the pipe.
# All files inside It will be deleted at the end of the treatment.
res_path = "/export/home/cse200093/Jacques_Bio/data_bio/test_measurment"


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
