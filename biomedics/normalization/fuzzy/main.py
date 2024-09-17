from collections import defaultdict
from pathlib import Path
from typing import Union

import duckdb
import pandas as pd
import edsnlp
from edsnlp.connectors import BratConnector
from unidecode import unidecode

from .exception import exception_list


class FuzzyNormalizer:
    def __init__(
        self,
        df_path: Union[str, Path],
        drug_dict: Union[dict, pd.DataFrame],
        label_to_normalize: str,
        with_qualifiers,
        atc_len=7
    ):
        if str(df_path).endswith("json"):
            self.df = pd.read_json(df_path)
            if "term_to_norm" not in self.df.columns:
                self.df["term_to_norm"] = self.df.term.str.lower().str.strip()
        else:
            self.df = self._gold_generation(
                df_path, label_to_normalize, with_qualifiers
            )

        self.unashable_cols = []
        for col in self.df.columns:
            if self.df[col].apply(lambda x: type(x) in [set, list]).sum() > 0:
                self.unashable_cols.append(col)
                self.df[col] = self.df[col].astype(str)
        self.df["term_to_norm"] = self.df["term_to_norm"].apply(lambda x: unidecode(x))

        self.drug_dict = self._make_drug_dict(drug_dict, atc_len)


    def _make_drug_dict(self, drug_dict, atc_len)->pd.DataFrame:
        if isinstance(drug_dict, dict):
            merged_dict = defaultdict(set)
            for atc_code, values in drug_dict.items():
                # Shorten the ATC code
                shortened_code = atc_code[:atc_len]
                merged_dict[shortened_code] |= set(values)

            merged_dict = (
                pd.DataFrame.from_dict({"norm_term": merged_dict}, "index")
                .T.explode("norm_term")
                .reset_index()
                .rename(columns={"index": "label"})
            )
            merged_dict.norm_term = merged_dict.norm_term.str.split(",")
            merged_dict = merged_dict.explode("norm_term").reset_index(drop=True)

        elif isinstance(drug_dict, pd.DataFrame):
            if "norm_term" not in self.df.columns:
                drug_dict.columns = ["label", "norm_term"]
                drug_dict = drug_dict.explode("norm_term")

            drug_dict.norm_term = drug_dict.norm_term.str.lower().str.strip()
            merged_dict = drug_dict

        else:
            raise ValueError(f"""
                Expected a dict of pd.DataFrame input, got {type(drug_dict)}
            """)

        return merged_dict

    def _gold_generation(self, df_path, label_to_normalize, with_qualifiers):
        """
        Generate a dataframe from a brat folder with qualifiers within the list below.
        """
        qualifiers = ["Temporality", "Certainty", "Action", "Negation"]
        doc_list = BratConnector(df_path).brat2docs(edsnlp.blank("eds"))
        ents_list = []
        for doc in doc_list:
            if label_to_normalize in doc.spans.keys():
                for ent in doc.spans[label_to_normalize]:
                    if hasattr(ent._, "Tech") and ent._.Tech:
                        continue
                    ent_data = [
                        ent.text,
                        doc._.note_id + ".ann",
                        (ent.start_char, ent.end_char),
                        ent.text.lower().strip(),
                    ]
                    if with_qualifiers:
                        for qualifier in qualifiers:
                            ent_data.append(getattr(ent._, qualifier))
                        ents_list.append(ent_data)
        df_columns = ["term", "source", "span_converted", "term_to_norm"]
        if with_qualifiers:
            df_columns += qualifiers

        df = pd.DataFrame(
            ents_list, columns=df_columns
        )
        return df

    def normalize(self, method: str = "lev", threshold: Union[int, float] = 10):

        df = self.df.copy()
        for index, row in df.iterrows():
            for k, v in exception_list.items():
                if row["term_to_norm"] in v:
                    df.at[index, "term_to_norm"] = k

        if method == "exact":
            df = df.merge(
                self.drug_dict, how="left", left_on="term_to_norm", right_on="norm_term"
            )

        elif method == "lev":
            df_2 = self.drug_dict.copy()
            merged_df = duckdb.query(
                "select *, levenshtein(df.term_to_norm, df_2.norm_term) score " \
                f"from df, df_2 where score < {threshold}"
            ).to_df()
            merged_df["term_to_norm_len"] = merged_df.term_to_norm.str.len()
            merged_df["norm_term_len"] = merged_df.norm_term.str.len()
            merged_df["max_len"] = merged_df[
                ["norm_term_len", "term_to_norm_len"]
            ].max(axis=1)

            merged_df["score"] =  1 - (merged_df["score"]/merged_df["max_len"])
            merged_df = merged_df.drop(columns=[
                    "norm_term_len",
                    "term_to_norm_len",
                    "max_len"
                ]
            )
            idx = (
                merged_df.groupby(["source", "span_converted"])[
                    "score"
                ].transform(max)
                == merged_df["score"]
            )
            merged_df = merged_df[idx]
            merged_df = df.merge(merged_df, on=list(df.columns), how="left")
            df = merged_df

        elif method == "jaro_winkler":
            df_2 = self.drug_dict.copy()  # noqa: F841
            merged_df = duckdb.query(
                "select *, jaro_winkler_similarity(df.term_to_norm, df_2.norm_term) score " \
                f"from df, df_2 where score > {threshold}"
            ).to_df()
            
            merged_df["span_converted"] = merged_df["span_converted"].apply(tuple)
            merged_df["Negation"] = merged_df["Negation"].apply(lambda x: x=="True")
            
            idx = (
                merged_df.groupby(["source", "span_converted"])[
                    "score"
                ].transform(max)
                == merged_df["score"]
            )
            merged_df = merged_df[idx]
            merged_df = df.merge(merged_df, on=list(df.columns), how="left")
            df = merged_df


        else:
            raise ValueError("""The Method selected is not implemented.
            The value should be among: 'exact', 'lev' or 'jaro_winkler'.""")

        df = df.groupby(
            list(df.columns.difference({"label", "norm_term"})), as_index=False, dropna=False  # noqa: E501 # type: ignore
        ).agg({"label": list, "norm_term": set}) # type: ignore

        for col in self.unashable_cols:
            df[col] = df[col].apply(lambda x: eval(x))
        return df

