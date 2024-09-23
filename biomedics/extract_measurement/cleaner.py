import re
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from biomedics.extract_measurement.bio_lexical_variant import (
    lexical_var_non_digit_values,
)


def _clean_lexical_variant(lex_var: Any) -> str:
    """
    Clean lexical variant to remove pollution
    """
    lex_var_str = str(lex_var).strip()

    clean_lex_var = re.sub(r"\u2026", " ", lex_var_str)  # remove ...
    clean_lex_var = re.sub(r"\\n", " ", clean_lex_var)  # remove "\n"
    clean_lex_var = re.sub(r"\n", " ", clean_lex_var)
    clean_lex_var = re.sub(r"\best\b", " ", clean_lex_var)  # remove "est"

    clean_lex_var = re.sub(r"[\:\\]", " ", clean_lex_var)

    clean_lex_var = re.sub(r"\bcomprise?\b", " ", clean_lex_var)
    clean_lex_var = re.sub(r"\bavec\b", "", clean_lex_var)
    clean_lex_var = re.sub(r"\bpour\b", "", clean_lex_var)
    clean_lex_var = re.sub(r"\bsoit\b", "", clean_lex_var)
    clean_lex_var = re.sub(r"\bdans\b", "", clean_lex_var)
    clean_lex_var = re.sub(r"\bsans\b", "", clean_lex_var)
    clean_lex_var = re.sub(r"\bselon\b", " ", clean_lex_var)

    clean_lex_var = re.sub(r'\s*\/\s*', '/', clean_lex_var)
    clean_lex_var = re.sub(r'rapp?orts?', ' ', clean_lex_var)
    clean_lex_var = re.sub(r'dosages?', ' ', clean_lex_var)
    clean_lex_var = re.sub(r'numerations?', ' ', clean_lex_var)
    clean_lex_var = re.sub(r'\bratio?\b', ' ', clean_lex_var)
    clean_lex_var = re.sub('µ', 'micro', clean_lex_var)

    return clean_lex_var

def _replace_lexical_var(value: str) -> str:
    """
    Remove non digit values in from the lexical variant
    Terms are reported in the dict lexical_var_non_digit_values
    """
    for key, regex in lexical_var_non_digit_values.items():
        value = re.sub(regex, key, str(value))
    return value


def _clean_value(value: Any) -> Optional[Union[float, str]]:
    """
    Clean value with pollution
    """
    pattern = r"[^\d.,/<>]+"
    value_cleaned = re.sub(pattern, "", str(value))

    match = re.findall(
        r'([<>]?\d+[\.\,]?\d*\/\d+[\.\,]?\d*|[<>]?\d+[\.\,]?\d*)$',
        value_cleaned
    )
    match_unclassic_format = re.findall(r'\b(0\d+)\b', value_cleaned)

    if len(match) > 0:
        val = match[0]
        if "/" in val or ">" in val or "<" in val:
            return value_cleaned
        if "," in val:
            val = val.replace(",", ".")
        if val.count(".") > 1:
            val = re.sub(r"\.{1,10}", ".", val)
        try:
            return float(val)
        except ValueError:
            return np.nan
    elif len(match_unclassic_format) > 0:
        return np.nan
    elif str(value).lower() == 'nan':
        return np.nan
    else:
        return value_cleaned


def _clean_unit(lex_var: Any) -> str:
    """
    Clean unit
    """
    clean_unit = re.sub(r"gramm?es?", "g", str(lex_var))
    clean_unit = re.sub(r"micr?o?", "µ", clean_unit)
    clean_unit = re.sub(r"micor", "µ", clean_unit)
    clean_unit = re.sub(r"moles?", "mol", clean_unit)
    clean_unit = re.sub(r"molaires?", "mol", clean_unit)
    clean_unit = re.sub(r"mill?i", "m", clean_unit)
    clean_unit = re.sub(r"litres?", "l", clean_unit)
    clean_unit = re.sub(r"secondes?", "sec", clean_unit)
    clean_unit = re.sub(r"secs?", "sec", clean_unit)
    clean_unit = re.sub(r"jours?", "j", clean_unit)
    clean_unit = re.sub(r"par", "/", clean_unit)
    clean_unit = re.sub(r"minutes?", "min", clean_unit)
    clean_unit = re.sub(r"heures?", "h", clean_unit)
    clean_unit = re.sub(r"\_", "", clean_unit)
    clean_unit = re.sub(r"\n", "", clean_unit)
    clean_unit = re.sub(r"\ℓ", "l", clean_unit)
    return clean_unit


def _normalise_unit(unit: Any) -> str:
    """
    Normalise units
    """
    unit_nospace = str(unit).replace(" ", "").lower()

    clean_unit_stripped = unit_nospace.strip("-,.¦| ")

    match_perc = re.findall(r'%', str(clean_unit_stripped))
    match_24 = re.findall(r'\/24\w\b', str(clean_unit_stripped))
    match_x10E = re.findall(r'x?10\*\d+\/[a-z]{1,2}', str(clean_unit_stripped))
    match_dfg_unit = re.findall(
        r'(ml\s?\/\s?mi?n\s?\/\s?1[.,]73\s?mml\s?\/\s?mi?n\s?\/\s?m2)',
        str(clean_unit_stripped)
    )
    if len(match_perc) > 0:
        return '%'
    elif len(match_x10E) > 0:
        return str(clean_unit_stripped)
    elif len(match_24) > 0:
        return clean_unit_stripped
    elif len(match_dfg_unit) > 0:
        return clean_unit_stripped
    else:
        return clean_unit_stripped


def normalize_expression(expression: str) -> str:
    return ' '.join(sorted(expression.split()))


def _convert_brat_spans(span: str) -> List[Optional[int]]:
    span_match = re.compile(r"^(\d+).*\s(\d+)$").match(span)
    try:
        span_start = int(span_match.group(1)) # type: ignore
        span_end = int(span_match.group(2)) # type: ignore
    except AttributeError:
        print("No span found.")
        return [None, None]
    return [span_start, span_end]

def match_bio_to_biocomp(
    df_bio: pd.DataFrame,
    df_biocomp: pd.DataFrame
) -> pd.DataFrame:
    """
    Match bio to biocomp entities retrieved from the EDS-Biomedic model and clean the
    lexical variants for pollution.
    """
    df_biocomp = df_biocomp.rename(columns={
        "lexical_variant": "lexical_variant_biocomp",
        "term": "term_biocomp"
    })

    # Rename df_bio columns
    df_bio = df_bio.rename(columns={col: f"{col}_bio" for col in df_bio.columns})

    # Perform the join
    df_biocomp_bio = pd.merge(
        df_biocomp,
        df_bio,
        left_on="source",
        right_on="source_bio",
        how="left"
    ).drop_duplicates()
    df_biocomp_bio =  df_biocomp_bio[
        (df_biocomp_bio.span_start <= df_biocomp_bio.span_start_bio)
        & (df_biocomp_bio.span_end >= df_biocomp_bio.span_end_bio)
    ]
    pattern_ellipse = r'((\.\.\.|…)\s?([¹²³⁰⁴⁵⁶⁷⁸⁹0-9]))'

    df_biocomp_bio["lexical_variant_biocomp"] = (
        df_biocomp_bio["lexical_variant_biocomp"]
        .str.replace(pattern_ellipse, ' ', regex=True)
    )
    df_biocomp_bio["lexical_variant_bio"] = (
        df_biocomp_bio["lexical_variant_bio"]
        .str.replace(pattern_ellipse, ' ', regex=True)
    )
    df_biocomp_bio["lexical_variant_biocomp"] = (
        df_biocomp_bio["lexical_variant_biocomp"]
        .str.replace('…', ' ').str.replace(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]", ' ', regex=True)
    )
    df_biocomp_bio["lexical_variant_bio"] = (
        df_biocomp_bio["lexical_variant_bio"]
        .str.replace('…', ' ').str.replace(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]", ' ', regex=True)
    )

    return df_biocomp_bio


def match_date_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return date if identified in the lexical_variant.
    If a date format is identified it is assigned to a new column `extracted_date`
    and remove the date in the lexical_variant.
    """
    date_pattern = (
        r"(\d{2}/20\d{2}|\d{2}/\d{2}|20\d{2}-\d{2}-\d{2}|"
        r"20\d{2}-\d{2}|20\d{2})"
    )

    df_with_dates = df.copy()
    df_with_dates["extracted_date"] = df_with_dates[
        "lexical_variant_biocomp"
    ].str.extract(date_pattern, expand=False)
    df_with_dates["lexical_variant_stripped"] = df_with_dates[
        "lexical_variant_stripped"
    ].str.replace(date_pattern, "", regex=True)

    return df_with_dates


def extract_clean_range_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return range value if identified in the lexical_variant.
    If a range value format is identified it is assigned to a new column `range_value`
    and removed in the lexical_variant.
    """
    pattern_range_value = (
        r"([|¦][<>]\d+[\.,]?\d*$|"
        r"\(\s?[nN]?\s?:?\s?[<>]\s?\d+[\.\,]?\d*\s?\)?|"
        r"\(?\s?[nN]?\s?[:=]?\s?\d+[\.,]?\d*\s?[\-–]\s?\d+[\.,]?\d*\s?\)?)"
    )

    df_range_val = df.copy()
    df_range_val["range_value"] = df_range_val[
        "lexical_variant_stripped"
    ].str.extract(pattern_range_value, expand=False)
    df_range_val["lexical_variant_stripped"] = df_range_val[
        "lexical_variant_stripped"
    ].str.replace(pattern_range_value, " ", regex=True)
    df_range_val["range_value"] = df_range_val["range_value"].str.replace(
        r"[^\d><\-–\.,]", "", regex=True
    )
    # Split the range into min and max
    split_range = df_range_val["range_value"].str.split(r"[\-–]", expand=True)
    df_range_val["range_min"] = split_range[0].replace(",", ".")
    df_range_val["range_max"] = split_range[1].replace(",", ".")

    # Convert range min and max to numeric, coercing errors to NaN
    df_range_val["range_min"] = pd.to_numeric(df_range_val["range_min"], errors="coerce")
    df_range_val["range_max"] = pd.to_numeric(df_range_val["range_max"], errors="coerce")

    return df_range_val


def clean_lexical_variant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean lexical variant to prepare for units and values extraction
    """
    df_clean = df.copy()
    df_clean["lexical_variant_stripped"] = df_clean[
        "lexical_variant_stripped"
    ].str.replace(r'⁹', " ", regex=True)
    df_clean["lexical_variant_stripped"] = df_clean[
        "lexical_variant_stripped"
    ].str.replace(r'm²', "m2", regex=True)
    df_clean["lexical_variant_stripped"] = df_clean[
        "lexical_variant_stripped"
    ].str.replace(r'm³', "m3", regex=True)
    df_clean["lexical_variant_stripped"] = df_clean[
        "lexical_variant_stripped"
    ].str.lower()
    translation_table = str.maketrans(
        '⁰¹²³⁴⁵⁶⁷⁸⁹',
        '0123456789'
    )
    df_clean["lexical_variant_stripped"] = df_clean[
        "lexical_variant_stripped"
    ].str.translate(translation_table) # type: ignore
    df_clean["lexical_variant_stripped"] = df_clean[
        "lexical_variant_stripped"
    ].str.replace(r'à', "=", regex=True)
    df_clean["lexical_variant_stripped"] = df_clean[
        "lexical_variant_stripped"
    ].str.replace(r'(\:)', " ", regex=True)
    df_clean["lexical_variant_stripped"] = df_clean[
        "lexical_variant_stripped"
    ].str.translate(
        str.maketrans(
            'ãäöüáàäčçďéêěèïíîĺľňóôŕšťúûůýž',
            'aaouaaaccdeeeeiiillnoorstuuuyz'
        ) # type: ignore
    )

    # Apply _clean_lexical_variant function
    df_clean["lexical_variant_stripped"] = df_clean[
        "lexical_variant_stripped"
    ].apply(_clean_lexical_variant)

    return df_clean


def extract_clean_non_digit_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return non digit value if identified in the lexical_variant.
    If a non digit value format is identified it is assigned to a new column
    `non_digit_value` and removed in the lexical_variant.
    """
    df_non_digit_val = df.copy()
    df_non_digit_val["lexical_variant_stripped"] = (
        df_non_digit_val["lexical_variant_stripped"]
        .apply(_replace_lexical_var)
    )

    pattern_non_digit_val = '|'.join(lexical_var_non_digit_values.values())

    df_non_digit_val["non_digit_value"] = df_non_digit_val[
        "lexical_variant_stripped"
    ].str.findall(pattern_non_digit_val)
    df_non_digit_val = df_non_digit_val.explode(["non_digit_value"])
    
    df_non_digit_val["non_digit_value"] = df_non_digit_val[
        "non_digit_value"
    ].str.replace(r"[^\w+-]", "", regex=True)
    df_non_digit_val["lexical_variant_stripped"] = df_non_digit_val[
        "lexical_variant_stripped"
    ].str.replace(
        '(' + pattern_non_digit_val + ')',
        " ",
        regex=True
    )

    df_non_digit_val["non_digit_value"] = df_non_digit_val[
        "non_digit_value"
    ].replace("", np.nan)

    return df_non_digit_val


def extract_clean_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return unit if identified in the lexical_variant.
    If a unit format is identified, it is assigned to a new column `unit` and removed in
    the lexical_variant.
    """
    df_units = df.copy()
    df_units["lexical_variant_stripped"] = (
        df_units["lexical_variant_stripped"]
        .apply(_clean_unit)
    )

    pattern_10E = r'(?i)(\d+(?:\.\d+)?)(x?10\*\d+\/[A-Za-z]+)'

    # Replace pattern_10E with '\1 \2'
    df_units["lexical_variant_stripped"] = df_units[
        "lexical_variant_stripped"
    ].str.replace(
        pattern_10E,
        r'\1 \2',
        regex=True
    )

    pattern_digit_nospace_wt_unit = (
        r"(?i)(\d+)(g?\/mm3|ml\s?\/\s?mi?n\s?\/\s?m2|ml\s?\/\s?mi?n\s?\/\s?1[.,]73\s?m2|"
        r"ml\s?\/\s?mi?n\s?\/\s?mi?n|µ?m?n?p?g\/24h|"
        r"µ?u?m?n?p?g\s?\/\s?d?m?l|m?mol/24h|seco?n?d?e?s?|"
        r"minu?t?e?s?|µ?u?m?n?p?g\s?\/\s?mmol?|"
        r"µ?u?m?n?p?mol\s?\/\s?[a-z0-9µ]+|"
        r"µ?u?m?n?p?g\s?\/\s?\w+?|mm\s?h\s?g\.?|µ?u?m?n?p?mol|"
        r"\%|fl\b|mgy\.cm|gigas?\s*\/?\w+?|"
        r"µ?u?m?n?p?g\b|\/?mm\b|gl\b|mil?li\s*grammes?|"
        r"grammes?\s*\/\s*\w*|micro\s*mol\/?\w*|\/?p?g?ml|m?osmol\/?\w*|"
        r"molaires?|u\s*\/\s*l|m?ui\s*\/\s*m?l|k?pa\b|"
        r"[a-zA-Zµ]+\/\w+|µ?u?m?n?m\b|[|¦]µ?u?m?n?s\b|\/[a-zA-Zµ]+\b)"
    )

    df_units["lexical_variant_stripped"] = df_units[
        "lexical_variant_stripped"
    ].str.replace(
        pattern_digit_nospace_wt_unit,
        r'\1 \2',
        regex=True
    )

    units_pattern = (
        r"(?i)(x?10\*\d+\s?\/[a-z]{1,2}|g?\/mm3|ml\s?\/\s?mi?n\s?\/\s?m2|"
        r"ml\s?\/\s?mi?n\s?\/\s?1[.,]73\s?m2|ml\s?\/\s?mi?n|"
        r"μ?µ?m?n?p?g\/24h|µ?u?m?n?p?g\s?\/\s?d?m?l|"
        r"m?mol/24h|seco?n?d?e?s?|\bminu?t?e?s?|"
        r"µ?u?m?n?p?g\s?\/\s?mmol?|μ?µ?u?m?n?p?mol\s?\/\s?[a-z0-9µ]+|"
        r"μ?µ?u?m?n?p?g\s?\/\s?\w+|mm\s?h\s?g\.?|μ?µ?u?m?n?p?mol|\%|\bfl\b|"
        r"mgy\.cm|gigas?\s*\/?\w+|\bμ?µ?u?m?n?p?g\b|\/?mm\b|gl\b|mil?li\s*grammes?|"
        r"grammes?\s*\/\s*\w*|micro\s*mol\/?\w*|\/?p?g?ml|m?osmol\/?\w*|"
        r"molaires?|u\s*\/\s*l|m?ui\s*\/\s*m?l|\bk?pa\b|"
        r"[a-zA-Zµμμ]+\/\w+|\bµ?u?m?n?m\b|"
        r"[|¦]μ?µ?u?m?n?s\b|[\s|¦]\/[a-zA-Zµμ]+\b)"
    )

    # Extract units
    df_units["unit"] = df_units[
        "lexical_variant_stripped"
    ].str.extract(
        units_pattern,
        flags=re.IGNORECASE,
        expand=False
    )
    # Remove the units from 'lexical_variant_stripped'
    df_units["lexical_variant_stripped"] = df_units[
        "lexical_variant_stripped"
    ].str.replace(
        units_pattern,
        " ",
        flags=re.IGNORECASE,
        regex=True
    )

    # Normalize units
    df_units["unit"] = df_units["unit"].apply(_normalise_unit)

    # Replace empty strings with NaN
    df_units["unit"] = df_units["unit"].replace("", np.nan)

    return df_units


def extract_clean_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return value if identified in the lexical_variant.
    If a value format is identified, it is assigned to a new column `value` and removed
    in the lexical_variant.
    Value is then cleaned and assigned to a new column `value_cleaned`.
    """
    pattern_values = (
        r"([<>]?\s?\d+[\.\,]?\s?\d*\/\d+[\.\,]?\s?\d*|"
        r"[<>]?\s?\d+[\.\,]?\s?\d*|\d)"
    )

    # Extract 'value'
    df_values = df.copy()
    df_values["value"] = df_values["lexical_variant_stripped"].str.extract(
        pattern_values,
        expand=False
    )
    # Remove 'value' from 'lexical_variant_stripped'
    df_values["lexical_variant_stripped"] = df_values[
        "lexical_variant_stripped"
    ].str.replace(
        pattern_values,
        " ",
        regex=True
    )

    # Clean the extracted 'value'
    df_values["value_cleaned"] = df_values["value"].apply(_clean_value)

    # Replace empty strings with NaN in 'value_cleaned'
    df_values["value_cleaned"] = df_values["value_cleaned"].replace(
        "",
        np.nan
    )

    return df_values


def extract_fluids_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return fluid source the bio was measured in if identified in the lexical_variant.
    If a fluid format is identified, it is assigned to a new column `fluid_source` and
    removed in the lexical_variant.
    The fluid terms are normalised using fuzzy matching using the reference dictionary
    fluide_source.
    """
    pattern_fluids = (
        r"(?i)(sang\w*|urine\w*|serique\w*|plasma\w*|foetal\w*|capill?air\w*|"
        r"serum|urinaire\w*|\bur\b|selle\w*|vessie|veineux|veineuse|\blcr\b)"
    )

    df_fluids = df.copy()
    # Extract 'fluid_source'
    df_fluids["fluid_source"] = df_fluids["lexical_variant"].str.extract(
        pattern_fluids,
        expand=False
    )
    # Remove 'fluid_source' from 'lexical_variant'
    df_fluids["lexical_variant"] = df_fluids["lexical_variant"].str.replace(
        pattern_fluids,
        " ",
        regex=True
    )

    return df_fluids


def extract_clean_subsequent_lex_var(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return subsequent lexical variant after removing all the previous entities.
    Lexical variant is normalised using fuzzy matching and lexical variant dictionary
    bios_lexical_variant_fuzzy_dict if the similarity score is higher than threshold the
    lexical variant term associated is returned in the column `lexical_variant_term`,
    the associated key in the column `lexical_variant_key` and the similarity score in
    the column `max_fuzzy_score`.
    If the max similarity score is not higher than the threshold, then it returns the
    initial term.
    """
    pattern_sub_lexical_var = (
        r"(?i)(\b[A-Za-z]+(\.[A-Za-z]+)*\b)"
    )

    df_subs_lex_var = df.copy()
    # Determine which lexical_variant to use
    df_subs_lex_var["lexical_variant"] = np.where(
        df_subs_lex_var["lexical_variant_bio"] != "",
        df_subs_lex_var["lexical_variant_bio"],
        df_subs_lex_var["lexical_variant_stripped"].str.findall(
            pattern_sub_lexical_var,
        )
    )

    # Normalize 'lexical_variant'
    df_subs_lex_var["lexical_variant"] = df_subs_lex_var[
        "lexical_variant"
    ].str.lower()
    translation_table = str.maketrans(
        'ãäöüáàäčçďéêěèïíîĺľňóôřšťúûůýž',
        'aaouaaaccdeeeeiiillnoorstuuuyz'
    )
    df_subs_lex_var["lexical_variant"] = df_subs_lex_var[
        "lexical_variant"
    ].str.translate(translation_table) # type: ignore

    # Apply _clean_lexical_variant function
    df_subs_lex_var["lexical_variant"] = df_subs_lex_var[
        "lexical_variant"
    ].apply(_clean_lexical_variant)

    # Further cleaning
    df_subs_lex_var["lexical_variant"] = df_subs_lex_var[
        "lexical_variant"
    ].str.replace(
        r"\bsur\b",
        "/",
        regex=True
    )
    df_subs_lex_var["lexical_variant"] = df_subs_lex_var[
        "lexical_variant"
    ].str.replace(
        r"[<>-]",
        " ",
        regex=True
    )
    df_subs_lex_var["lexical_variant"] = df_subs_lex_var[
        "lexical_variant"
    ].str.strip()

    # Extract fluid sources
    df_subs_lex_var = extract_fluids_source(df_subs_lex_var)

    return df_subs_lex_var
