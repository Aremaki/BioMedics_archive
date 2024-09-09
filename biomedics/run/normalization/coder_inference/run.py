import os
from argparse import ArgumentParser
from pathlib import Path

import edsnlp
import pandas as pd
import submitit
from edsnlp.connectors import BratConnector
from omegaconf import OmegaConf

from biomedics.normalization.coder_inference.main import coder_wrapper

os.environ["OMP_NUM_THREADS"] = "16"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the model")
    parser.add_argument("--input_dir", type=Path, required=True, help="Path to the input directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Path to the output directory")
    parser.add_argument("--config", type=Path, required=True, help="Path to the config file")
    return parser.parse_args()

def coder_inference_cli(
    model_path: Path,
    input_dir: Path,
    output_dir: Path,
    config: Path,
):
    # Load the config file using OmegaConf
    config = OmegaConf.load(config)

    if str(input_dir).endswith(".pkl"):
        df = pd.read_pickle(input_dir)
        if config.column_name_to_normalize not in df.columns:
            replacement_col = (
                "terms_linked_to_measurement"
                if "terms_linked_to_measurement" in df.columns
                else "term"
            )
            df = df.explode(replacement_col)
            df[config.column_name_to_normalize] = df[replacement_col]
            # what if term is not in the columns of the df?
    else:
        doc_list = BratConnector(input_dir).brat2docs(edsnlp.blank("eds"))
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
        df_columns = [
            "term",
            "source",
            "span_converted",
            config.column_name_to_normalize,
        ] + config.qualifiers

        df = pd.DataFrame(
            ents_list, columns=df_columns
        )
    df = df[~df[config.column_name_to_normalize].isna()]
    df = coder_wrapper(df, config, model_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path_file = output_dir / f"{input_dir.stem}.json"
    df.to_json(path_file)


def main():
    args = parse_args()
    executor = submitit.AutoExecutor(folder=args.output_dir)
    executor.submit(
        coder_inference_cli,
        args.model_path,
        args.input_dir,
        args.output_dir,
        args.config,
    )

if __name__ == "__main__":
    main()
