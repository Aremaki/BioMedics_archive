from typing import Dict, List, Union

from edsnlp.pipelines.misc.measurements.measurements import (
    MeasureConfig,
    MeasurementsMatcher,
    UnitConfig,
)
from spacy.language import Language

from . import patterns as patterns

DEFAULT_CONFIG = dict(
    attr="NORM",
    ignore_excluded=True,
    units_config=patterns.units_config,
    number_terms=patterns.number_terms,
    unit_divisors=patterns.unit_divisors,
    measurements=patterns.measurements,
    stopwords=patterns.stopwords,
    merge_mode="intersect",
)


@Language.factory("doses", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    measurements: Union[
        List[Union[str, MeasureConfig]],
        Dict[str, MeasureConfig],
    ],
    units_config: Dict[str, UnitConfig],
    number_terms: Dict[str, List[str]],
    stopwords: List[str],
    unit_divisors: List[str],
    ignore_excluded: bool,
    attr: str,
    merge_mode: str = "intersect",
):
    return MeasurementsMatcher(
        nlp,
        name="doses",
        units_config=units_config,
        number_terms=number_terms,
        unit_divisors=unit_divisors,
        measurements=measurements,
        stopwords=stopwords,
        attr=attr,
        ignore_excluded=ignore_excluded,
        as_ents=True,
        merge_mode=merge_mode,
    )
