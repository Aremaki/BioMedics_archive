import os

os.environ["OMP_NUM_THREADS"] = "16"
from extract_measurements_from_brat import ExtractMeasurements
from config_part_2 import *

import pandas as pd


def main():
    # Measurements
    df = ExtractMeasurements(
        regex_convert_spans = measurements_pipe_regex_convert_spans,
        label_key = measurements_pipe_label_key,
        labels_to_remove = measurements_pipe_labels_to_remove,
        labels_linkable_to_measurement = measurements_pipe_labels_linkable_to_measurement,
        config_normalizer_from_label_key = measurements_pipe_config_normalizer_from_label_key,
        config_measurements_from_label_key = measurements_pipe_config_measurements_from_label_key,
        config_normalizer_from_tables = measurements_pipe_config_normalizer_from_tables,
        config_measurements_from_tables = measurements_pipe_config_measurements_from_tables,
    )(brat_dir = measurements_pipe_brat_dir, only_tables = measurements_only_tables)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    df.to_json(res_path + "/res.json")

main()
