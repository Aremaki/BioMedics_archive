import os
import pickle
from pathlib import Path

import pandas as pd
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from .get_normalization_with_coder import CoderNormalizer
from .text_preprocessor import TextPreprocessor

os.environ["OMP_NUM_THREADS"] = "16"

def coder_wrapper(
        df: pd.DataFrame,
        config: DictConfig | ListConfig,
        model_path: str | Path
    ):
    # This wrapper is needed to preprocess terms
    # and in case the cells contains list of terms instead of one unique term
    df = df.reset_index(drop=True)
    text_preprocessor = TextPreprocessor(
        cased=config.coder_cased,
        stopwords=config.coder_stopwords
    )

    coder_normalizer = CoderNormalizer(
        model_name_or_path=model_path,
        tokenizer_name_or_path=model_path,
        device=config.coder_device,
    )

    # Preprocess UMLS
    print("--- Preprocessing UMLS ---")
    if str(config.umls_path).endswith(".json"):
        umls_df = pd.read_json(config.umls_path)
    elif str(config.umls_path).endswith(".pkl"):
        umls_df = pd.read_pickle(config.umls_path)
        umls_df = umls_df.explode(config.synonyms_column_name)
    else:
        raise ValueError("umls_path should be a json or pkl file.")
    
    umls_df[config.synonyms_column_name] = umls_df[config.synonyms_column_name].apply(
        lambda term: text_preprocessor(
            text=term,
            remove_stopwords=config.coder_remove_stopwords_umls,
            remove_special_characters=config.coder_remove_special_characters_umls,
        )
    )
    umls_df = (
        umls_df.loc[
            (~umls_df[config.synonyms_column_name].str.isnumeric())
            & (umls_df[config.synonyms_column_name] != "")
        ]
        .groupby([config.synonyms_column_name])
        .agg({config.labels_column_name: set, config.synonyms_column_name: "first"})
        .reset_index(drop=True)
    )
    coder_umls_des_list = umls_df[config.synonyms_column_name]
    coder_umls_labels_list = umls_df[config.labels_column_name]
    if config.coder_save_umls_des_dir:
        with open(config.coder_save_umls_des_dir, "wb") as f:
            pickle.dump(coder_umls_des_list, f)
    if config.coder_save_umls_labels_dir:
        with open(config.coder_save_umls_labels_dir, "wb") as f:
            pickle.dump(coder_umls_labels_list, f)

    # Preprocessing and inference on terms
    print("--- Preprocessing terms ---")

    exploded_term_df = (
        pd.DataFrame(
            {
                "id": df.index,
                config.column_name_to_normalize: df[config.column_name_to_normalize],
            }
        )
        .explode(config.column_name_to_normalize)
        .reset_index(drop=True)
    )
    coder_data_list = (
        exploded_term_df[config.column_name_to_normalize]
        .apply(
            lambda term: text_preprocessor(
                text=term,
                remove_stopwords=config.coder_remove_stopwords_terms,
                remove_special_characters=config.coder_remove_special_characters_terms,
            )
        )
        .tolist()
    )

    print("--- CODER inference ---")
    coder_res = coder_normalizer(
        umls_labels_list=coder_umls_labels_list,
        umls_des_list=coder_umls_des_list,
        data_list=coder_data_list,
        save_umls_embeddings_dir=config.coder_save_umls_embeddings_dir,
        save_data_embeddings_dir=config.coder_save_data_embeddings_dir,
        normalize=config.coder_normalize,
        summary_method=config.coder_summary_method,
        tqdm_bar=config.coder_tqdm_bar,
        coder_batch_size=config.coder_batch_size,
    )
    exploded_term_df[
        ["label", "norm_term", "score"]
    ] = pd.DataFrame(zip(*coder_res))
    df = (
        pd.merge(
            df.drop(columns=[config.column_name_to_normalize]),
            exploded_term_df,
            left_index=True,
            right_on="id",
        )
        .drop(columns=["id"])
        .reset_index(drop=True)
    )
    return df
