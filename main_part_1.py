from get_ner_with_eds_medic import EdsMedicNer
from config_part_1 import *

import pandas as pd


def main():
    # NER
    EdsMedicNer(
        model_path=eds_medic_model_path,
        gpu_id=eds_medic_gpu_id,
        silent=eds_medic_silent
    )(data_path, eds_medic_output_path)


main()
