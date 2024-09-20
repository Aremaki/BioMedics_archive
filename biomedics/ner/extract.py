import os
from typing import List, Optional, Union

import edsnlp
import pandas as pd
import spacy
import torch
import typer
from spacy.tokens import Doc


def convert_doc_to_dict(doc: Doc, attributes: Optional[List[str]] = None) -> List[dict]:
    if attributes is None:
        attributes = []
    ents = [
        {**{
            "note_id": doc._.note_id,
            "lexical_variant": e.text,
            "label": e.label_,
            "start": e.start,
            "end": e.end,
        }, **{
            attr: getattr(e._, attr) for attr in attributes
        }}
        for e in doc.ents
    ]
    if "BIO" in doc.spans:
        spans = [
            {**{
                "note_id": doc._.note_id,
                "lexical_variant": s.text,
                "label": "BIO",
                "start": s.start,
                "end": s.end,
            }, **{
                attr: getattr(s._, attr) for attr in attributes
            }}
            for s in doc.spans["BIO"]
        ]
    else:
        spans = []
    return ents + spans

def build_data(
    corpus: Union[str, pd.DataFrame],
):
    """
    This function builds a data iterator from a text corpus.
    The data iterator can then be used to map a nlp model to the txts or for other
    functions.
    Args:
        - corpus: either a directory with txts inside, a path to a .csv file that is in
    the form ["note_id", "note_txt"] or a pandas DataFrame with the same columns.
    Returns:
        An iterator of spacy docs of the corpus.
    """
    if isinstance(corpus, str):
        if os.path.isdir(corpus):
            print(f"Building from dir {corpus}")
            data = edsnlp.data.read_standoff(corpus) # type: ignore

        elif corpus.endswith(".csv"):
            print(f"Loading as a csv from {corpus}")
            df = pd.read_csv(corpus)
            data = edsnlp.data.from_pandas(df, converter="omop") # type: ignore

        else:
            raise ValueError("The corpus must be a directory or a csv file")

    elif isinstance(corpus, pd.DataFrame):
        print("Using corpus as a pandas DataFrame")
        data = edsnlp.data.from_pandas(corpus, converter="omop") # type: ignore

    else:
        raise TypeError(f"Expected str of pd.DataFrame types, got {type(corpus)}")

    return data

def extract_ents_from_docs(
    docs,
    nlp: edsnlp.Pipeline,
    attributes: Optional[List[str]] = None
) -> pd.DataFrame:

    docs = docs.map_pipeline(nlp)

    if torch.cuda.is_available():
        print("Using GPU")
        docs = docs.set_processing(
            num_cpu_workers=4,
            num_gpu_workers=1,
            batch_size=32,
        )

    def converter_with_attributes(doc: Doc):
        return convert_doc_to_dict(doc, attributes=attributes)

    df_ents = edsnlp.data.to_pandas( # type: ignore
        docs,
        converter=converter_with_attributes,
    )

    return df_ents

def main(
    root: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    try:
        from biomedics.ner.loaders import eds_biomedic
        if torch.cuda.is_available():
            print("Using GPU")
            spacy.require_gpu()

        print("Using EDS-Biomedic")
        nlp = eds_biomedic()
    except ImportError:
        print("EDS-Biomedic not found, using default model")
        nlp = edsnlp.blank("fr")

    docs = build_data(root)
    basic_attributes = [
        "Negation",
        "Certainty",
        "Family",
        "Action",
        "Temporality",
    ]
    df_ents = extract_ents_from_docs(docs, nlp, basic_attributes)
    if output_path:
        df_ents.to_parquet(
            output_path + ".parquet",
            engine="pyarrow",
            compression="snappy"
        )

    return df_ents

if __name__ == "__main__":
    typer.run(main)
