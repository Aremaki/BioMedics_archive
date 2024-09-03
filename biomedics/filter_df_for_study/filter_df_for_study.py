from tqdm import tqdm
import pandas as pd
import typer
from confection import Config
from loguru import logger
from rich import print

def filter_bio_structured(config_anabio_codes):
    bio_result_folder_path = BASE_DIR / "data" / "bio_result"
    bio_from_structured_data = pd.read_pickle(bio_result_folder_path / "bio_from_structured_data.pkl")
    codes_to_keep = {"disease": [], "concept_cd": [], "bio": []}
    for disease, anabio_codes in config_anabio_codes.items():
        for label, code_list in anabio_codes.items():
            for code in code_list:
                codes_to_keep["disease"].append(disease)
                codes_to_keep["concept_cd"].append(f"LAB:{code}")
                codes_to_keep["bio"].append(label)
    filtered_bio = bio_from_structured_data.merge(
        pd.DataFrame(codes_to_keep), on=["disease", "concept_cd"]
    )
    for disease in config_anabio_codes.keys():
        path_to_res = bio_result_folder_path / disease
        if not os.path.exists(path_to_res):
            mkdir(path_to_res)
        filtered_bio[filtered_bio.disease == disease].to_pickle(path_to_res / "filtered_bio_from_structured_data.pkl")
    filtered_bio.to_pickle(bio_result_folder_path / "filtered_bio_from_structured_data.pkl")
    
def filter_med_structured(config_atc_codes):
    med_result_folder_path = BASE_DIR / "data" / "drug_result"
    med_from_structured_data = pd.read_pickle(med_result_folder_pat / "med_from_structured_data.pkl")
    codes_to_keep = {"disease": [], "valueflag_cd": [], "med": []}
    for disease, atc_codes in config_atc_codes.items():
        for label, code_list in atc_codes.items():
            for code in code_list:
                codes_to_keep["disease"].append(disease)
                codes_to_keep["valueflag_cd"].append(code)
                codes_to_keep["med"].append(label)
    filtered_med = med_from_structured_data.merge(
        pd.DataFrame(codes_to_keep), on=["disease", "valueflag_cd"]
    )
    med_from_structured_data["valueflag_cd"] = med_from_structured_data[
        "valueflag_cd"
    ].str.slice(stop=5)
    filtered_med_short = med_from_structured_data.merge(
        pd.DataFrame(codes_to_keep), on=["disease", "valueflag_cd"]
    )
    filtered_med = pd.concat([filtered_med, filtered_med_short])
    for disease in config_atc_codes.keys():
        path_to_res = med_result_folder_path / disease
        if not os.path.exists(path_to_res):
            mkdir(path_to_res)
        filtered_med[filtered_med.disease == disease].to_pickle(path_to_res / "filtered_med_from_structured_data.pkl")
    display(filtered_med)
    filtered_med.to_pickle(RES_DRUG_DIR / "filtered_med_from_structured_data.pkl")
    
def filter_bio_nlp(config_cui_codes):
    # List of df by disease for concatenation
    res_part_filtered_list = []
    for disease, cui_codes in config_cui_codes.items():
        ### Load each res dataset to concat them in one unique df
        bio_result_folder_path = BASE_DIR / "data" / "bio_result"
        res_part_df = pd.read_json(bio_result_folder_path / disease / "pred_bio_coder_all.json")
        res_part_df["disease"] = disease

        ### Filter CUIS to keep
        codes_to_keep = {"disease": [], "label": [], "bio": []}
        for label, code_list in cui_codes.items():
            for code in code_list:
                codes_to_keep["disease"].append(disease)
                codes_to_keep["label"].append(code)
                codes_to_keep["bio"].append(label)
        res_part_df = res_part_df.explode("label")
        res_part_filtered = res_part_df.merge(
            pd.DataFrame(codes_to_keep), on=["disease", "label"]
        )
        res_part_filtered = res_part_filtered.groupby(
            list(res_part_filtered.columns.difference(["label"])),
            as_index=False,
            dropna=False,
        ).agg({"label": list})

        ### Save for future concatenation
        res_part_filtered.to_pickle(bio_result_folder_path / disease / "filtered_bio_from_nlp.pkl")
        res_part_filtered_list.append(res_part_filtered)
    res_filtered_df = pd.concat(res_part_filtered_list)
    res_filtered_df.to_pickle(bio_result_folder_path / "filtered_bio_from_nlp.pkl")
    
def filter_med_nlp(config_atc_codes):
    # List of df by disease for concatenation
    res_part_filtered_list = []
    for disease, atc_codes in config_atc_codes.items():
        ### Load each res dataset to concat them in one unique df
        med_result_folder_path = BASE_DIR / "data" / "drug_result"
        res_part_df = pd.read_json(med_result_folder_path / disease / "pred_med_fuzzy_jaro_winkler.json")
        res_part_df["disease"] = disease
        res_part_df["instance_num"] = res_part_df["source"].str.slice(stop=-4)

        ### Filter ATC to keep
        codes_to_keep = {"disease": [], "label": [], "med": []}
        for label, code_list in atc_codes.items():
            for code in code_list:
                codes_to_keep["disease"].append(disease)
                codes_to_keep["label"].append(code)
                codes_to_keep["med"].append(label)
        res_part_df = res_part_df.explode("label")
        res_part_filtered = res_part_df.merge(
            pd.DataFrame(codes_to_keep), on=["disease", "label"]
        )
        res_part_filtered = res_part_filtered.groupby(
            list(res_part_filtered.columns.difference(["label"])),
            as_index=False,
            dropna=False,
        ).agg({"label": list})

        ### Save for future concatenation
        res_part_filtered.to_pickle(med_result_folder_path / disease / "filtered_med_from_nlp.pkl")
        res_part_filtered_list.append(res_part_filtered)
    res_filtered_df = pd.concat(res_part_filtered_list)
    res_filtered_df.to_pickle(med_result_folder_path / "res_final_filtered.pkl")
    
    
def main(config_name: str = "config.cfg"):
    # Load config
    config_path = BASE_DIR / "conf" / config_name
    config = Config().from_disk(config_path, interpolate=True)
    if config["debug"]["debug"]:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
if __name__ == "__main__":
    typer.run(main)