from get_ner_with_eds_medic import EdsMedicNer
from extract_measurements_from_brat import ExtractMeasurements
from utils.text_preprocessor import TextPreprocessor
from get_normalization_with_coder import CoderNormalizer
from config import *

import pickle
import pandas as pd

def coder_wrapper(df):
    # This wrapper is needed to preprocess terms
    # and in case the cells contains list of terms instead of one unique term
    df = df.reset_index(drop=True)
    text_preprocessor = TextPreprocessor(
        cased=coder_cased,
        stopwords=coder_stopwords
    )
    coder_normalizer = CoderNormalizer(
        model_name_or_path = coder_model_name_or_path,
        tokenizer_name_or_path = coder_tokenizer_name_or_path,
        device = coder_device
    )
    
    # Preprocess UMLS
    print("--- Preprocessing UMLS ---")
    umls_df = pd.read_json(umls_path)
    
    umls_df[synonyms_column_name] = umls_df[synonyms_column_name].apply(lambda term:
                                                                        text_preprocessor(
                                                                            text = term,
                                                                            remove_stopwords = coder_remove_stopwords_umls,
                                                                            remove_special_characters = coder_remove_special_characters_umls)
                                                                       )
    umls_df = (
        umls_df.loc[(~umls_df[synonyms_column_name].str.isnumeric()) & (umls_df[synonyms_column_name] != "")]
        .groupby([synonyms_column_name])
        .agg({labels_column_name: set, synonyms_column_name: "first"})
        .reset_index(drop=True)
    )
    coder_umls_des_list = umls_df[synonyms_column_name]
    coder_umls_labels_list = umls_df[labels_column_name]
    if coder_save_umls_des_dir:
        with open(coder_save_umls_des_dir, "wb") as f:
            pickle.dump(coder_umls_des_list, f)
    if coder_save_umls_labels_dir:
        with open(coder_save_umls_labels_dir, "wb") as f:
            pickle.dump(coder_umls_labels_list, f)
    
    # Preprocessing and inference on terms
    print("--- Preprocessing terms ---")
    if type(df[column_name_to_normalize].iloc[0]) == str:
        coder_data_list = df[column_name_to_normalize].apply(lambda term:
                                                            text_preprocessor(
                                                                text = term,
                                                                remove_stopwords = coder_remove_stopwords_terms,
                                                                remove_special_characters = coder_remove_special_characters_terms)
                                                            ).tolist()
        print("--- CODER inference ---")
        coder_res = coder_normalizer(
            umls_labels_list = coder_umls_labels_list,
            umls_des_list = coder_umls_des_list,
            data_list = coder_data_list,
            save_umls_embeddings_dir = coder_save_umls_embeddings_dir,
            save_data_embeddings_dir = coder_save_data_embeddings_dir,
            normalize = coder_normalize,
            summary_method = coder_summary_method,
            tqdm_bar = coder_tqdm_bar,
            coder_batch_size = coder_batch_size,
        )
        df[["label", "des"]] = pd.DataFrame(zip(*coder_res))
    else:
        exploded_term_df = pd.DataFrame({
            "id": df.index,
            column_name_to_normalize: df[column_name_to_normalize]
        }).explode(column_name_to_normalize).reset_index(drop=True)
        coder_data_list = exploded_term_df[column_name_to_normalize].apply(lambda term:
                                                                           text_preprocessor(
                                                                               text = term,
                                                                               remove_stopwords = coder_remove_stopwords_terms,
                                                                               remove_special_characters = coder_remove_special_characters_terms)
                                                                          ).tolist()
        print("--- CODER inference ---")
        coder_res = coder_normalizer(
            umls_labels_list = coder_umls_labels_list,
            umls_des_list = coder_umls_des_list,
            data_list = coder_data_list,
            save_umls_embeddings_dir = coder_save_umls_embeddings_dir,
            save_data_embeddings_dir = coder_save_data_embeddings_dir,
            normalize = coder_normalize,
            summary_method = coder_summary_method,
            tqdm_bar = coder_tqdm_bar,
            coder_batch_size = coder_batch_size,
        )
        exploded_term_df[["label", "des"]] = pd.DataFrame(zip(*coder_res))
        df = pd.merge(df.drop(columns=[column_name_to_normalize]), exploded_term_df, left_index = True, right_on = "id").drop(columns=["id"]).reset_index(drop=True)
    return df


def main():
    # NER
    if do_ner:
        EdsMedicNer(
            model_path=eds_medic_model_path,
            gpu_id=eds_medic_gpu_id,
            silent=eds_medic_silent
        )(data_path, eds_medic_output_path)
    # Measurements
    if do_measurements_pipe:
        df = ExtractMeasurements(
            regex_convert_spans = measurements_pipe_regex_convert_spans,
            label_key = measurements_pipe_label_key,
            labels_to_remove = measurements_pipe_labels_to_remove,
            labels_linkable_to_measurement = measurements_pipe_labels_linkable_to_measurement,
            config_normalizer_from_label_key = measurements_pipe_config_normalizer_from_label_key,
            config_measurements_from_label_key = measurements_pipe_config_measurements_from_label_key,
            config_normalizer_from_tables = measurements_pipe_config_normalizer_from_tables,
            config_measurements_from_tables = measurements_pipe_config_measurements_from_tables
        )(brat_dir = measurements_pipe_brat_dir, only_tables = measurements_only_tables)
    # Coder
    if do_coder:
        df = coder_wrapper(df)
    # Saving
    if res_path:
        df.to_json(res_path)

if __name__ == "__main__":
    main()
