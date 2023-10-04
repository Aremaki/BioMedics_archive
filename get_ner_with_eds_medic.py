import pandas as pd 
import numpy as np
import spacy
from edsnlp.connectors.brat import BratConnector
import re
import srsly
import typer
from spacy.scorer import Scorer

from spacy.tokens import Doc
from spacy.training import Example
from spacy import util
from spacy.cli._util import Arg, Opt, import_code, setup_gpu
from spacy.cli.evaluate import (
    print_prf_per_type,
    print_textcats_auc_per_cat,
    render_parses,
)

import re
from pathlib import Path
from typing import Any, Dict, Optional

import os
from spacy.tokens import DocBin
from thinc.api import fix_random_seed
from wasabi import Printer

class EdsMedicNer():
    def __init__(
        self,
        model_path,
        gpu_id=0,
        silent=False,
    ):
        setup_gpu(use_gpu = gpu_id, silent = silent)
        self.model = spacy.load(model_path)
    
    def ner(self, data_path, output_path):
        brat = BratConnector(data_path, attributes = {"negation":"negation", "hypothetique": "hypothetique", "non_associe":"non_associe"})
        nlp = spacy.blank("fr")
        df_gold = brat.brat2docs(nlp)
        df_gold.sort(key=lambda doc: doc.text)
        
        print('-- Model running --')
        df_txt = [doc.text for doc in df_gold]
        
        df_txt_inferred = []
        for i, doc in enumerate(df_txt):
            inferred = self.model(doc)
            inferred._.trf_data = None
            df_txt_inferred.append(inferred)
        df_txt = df_txt_inferred
        
        # Because inference stops when It encounters too many documents, we coded this loop
        # Which delete the Transformers data from docs.
        # Here is the equivalent if the transformer data was not saved by default:
        # df_txt = [self.model(doc) for doc in df_txt]

        for i in range(len(df_txt)):
            df_txt[i]._.note_id = df_gold[i]._.note_id
        print('-- try saving --')

        print('path: ', output_path)
        # Instantiate the connector
        brat = BratConnector(output_path, attributes = {"negation":"negation", "hypothetique": "hypothetique", "non_associe":"non_associe"})
        # Convert all BRAT files to a list of documents
        brat.docs2brat(df_txt)

        print('-- saved -- ')

    def __call__(self, data_path, output_path):
        self.ner(data_path, output_path)
