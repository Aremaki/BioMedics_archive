import os
os.environ["OMP_NUM_THREADS"] = "16"

import typer
import spacy

import pickle
from pathlib import Path

import pandas as pd
from biomedics.normaization.coder_inference.main import coder_wrapper
from edsnlp.connectors import BratConnector


def coder_inference_cli(
    model_path: Path,
    input_dir: Path,
    output_dir: Path,
    data_type: str,
):
    if data_type == "BIO":
        import bio_config as config
    elif data_type == "MED":
        import med_config as config
    if str(input_dir).endswith(".pkl"):
        df = pd.read_pickle(input_dir)
        if not config.column_name_to_normalize in df.columns:
            if "terms_linked_to_measurement" in df.columns:
                df = df.explode("terms_linked_to_measurement")
                df = df.rename(columns={"terms_linked_to_measurement": config.column_name_to_normalize})
            else:
                df[config.column_name_to_normalize] = df.term
    else:
        doc_list = BratConnector(input_dir).brat2docs(spacy.blank("eds"))
        ents_list = []
        for doc in doc_list:
            if config.label_to_normalize in doc.spans.keys():
                for ent in doc.spans[config.label_to_normalize]:
                    ent_data = [
                        ent.text,
                        doc._.note_id + ".ann",
                        [ent.start_char, ent.end_char],
                        ent.text.lower().strip(),
                    ]
                    for qualifier in config.qualifiers:
                        ent_data.append(getattr(ent._, qualifier))
                    ents_list.append(ent_data)
        df_columns = ["term", "source", "span_converted", config.column_name_to_normalize] + config.qualifiers
        df = pd.DataFrame(
            ents_list, columns=df_columns
        )
    df = df[~df[config.column_name_to_normalize].isna()]
    df = coder_wrapper(df, config, model_path)
    if not os.path.exists(output_dir.parent):
        os.makedirs(output_dir.parent)
    df.to_json(output_dir)


if __name__ == "__main__":
    typer.run(coder_inference_cli)
