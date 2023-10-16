from spacy.language import Language

from .frequencies import FrequencyMatcher

DEFAULT_CONFIG = dict(
    threshold=3.0,
)


@Language.factory("frequencies", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    threshold: float = 3,
    merge_mode: str = "union",
):
    return FrequencyMatcher(
        nlp,
        threshold=threshold,
        merge_mode=merge_mode,
    )
