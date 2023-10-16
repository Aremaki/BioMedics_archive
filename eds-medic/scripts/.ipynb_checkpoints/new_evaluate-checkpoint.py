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

from eds_medic.corpus_reader import Corpus




def evaluate_cli(
    model: str = Arg(..., help="Model name or path"),  # noqa: E501
    data_path: Path = Arg(..., help="Location of binary evaluation data in .spacy format", exists=True),  # noqa: E501
    output: Optional[Path] = Opt(None, "--output", "-o", help="Output JSON file for metrics", dir_okay=False),  # noqa: E501
    docbin: Optional[Path] = Opt(None, "--docbin", help="Output Doc Bin path", dir_okay=False),  # noqa: E501
    code_path: Optional[Path] = Opt(None, "--code", "-c", help="Path to Python file with additional code (registered functions) to be imported"),  # noqa: E501
    use_gpu: int = Opt(-1, "--gpu-id", "-g", help="GPU ID or -1 for CPU"),  # noqa: E501
    gold_preproc: bool = Opt(False, "--gold-preproc", "-G", help="Use gold preprocessing"),  # noqa: E501
    displacy_path: Optional[Path] = Opt(None, "--displacy-path", "-dp", help="Directory to output rendered parses as HTML", exists=True, file_okay=False),  # noqa: E501
    displacy_limit: int = Opt(25, "--displacy-limit", "-dl", help="Limit of parses to render as HTML"),  # noqa: E501
):
    
    evaluate(model,
        data_path = '../data/test_eval/true_2doc',
        output=output,
        docbin=docbin,
        use_gpu=use_gpu,
        gold_preproc=gold_preproc,
        displacy_path=displacy_path,
        displacy_limit=displacy_limit,
        silent=False,
    )

    
def evaluate(
    model: str,
    data_path: Path,
    output: Optional[Path] = None,
    docbin: Optional[Path] = None,
    use_gpu: int = -1,
    gold_preproc: bool = False,
    displacy_path: Optional[Path] = None,
    displacy_limit: int = 25,
    silent: bool = True,
    spans_key: str = "sc",
):
    
    
   
    setup_gpu(use_gpu, silent)

    brat = BratConnector(data_path)
    empty = spacy.blank("fr")
    df_gold = brat.brat2docs(empty)
    df_gold.sort(key=lambda doc: doc.text)


    df_txt = [doc.text for doc in df_gold]
    model = spacy.load(model)
    df_txt = [model(doc) for doc in df_txt]
  

    for i in range(len(df_txt)):
        s =[]
        print(df_gold[i]._.note_id)
        example = Example(df_txt[i], df_gold[i])
        s.append(example)
        scorer = Scorer()
        r = scorer.score(s)
        r = r['ents_per_type']
        print(data_path)
        for i in r.keys():
            r[i]['p'] = round(r[i]['p'], 5)
            r[i]['r'] = round(r[i]['r'], 5)
            r[i]['f'] = round(r[i]['f'], 5)
            print(i, r[i]['p'], r[i]['r'], r[i]['f'])
        
        print('--------------')

    print('--- Over ---')

    
if __name__ == "__main__":
    typer.run(evaluate_cli)
