import spacy
from edsnlp.matchers.simstring import (
    SimilarityMeasure,
    SimstringMatcher,
    get_text_and_offsets,
    similarity,
)
from edsnlp.pipelines.base import BaseComponent
from edsnlp.pipelines.ner.drugs.patterns import get_patterns


@spacy.Language.factory(
    "drugs_normalizer",
    default_config=dict(
        measure=SimilarityMeasure.dice,
        threshold=0.75,
        windows=5,
        ignore_excluded=False,
        attr="NORM",
    ),
    requires=["doc.ents", "doc.spans"],
    assigns=["token.ent_kb_id"],
    default_score_weights={
        "norm_f": 1.0,
    },
)
class DrugsNormalizer(BaseComponent):
    def __init__(
        self,
        nlp: spacy.Language,
        name: str = "drugs_normalizer",
        measure: SimilarityMeasure = SimilarityMeasure.dice,
        threshold: float = 0.75,
        windows: int = 5,
        ignore_excluded: bool = False,
        attr: str = "NORM",
    ):
        self.matcher = SimstringMatcher(
            vocab=nlp.vocab,
            measure=measure,
            threshold=threshold,
            windows=windows,
            ignore_excluded=ignore_excluded,
            attr=attr,
        )
        self.matcher.build_patterns(
            nlp=nlp,
            terms=get_patterns(),
        )
        self.set_extensions()

    def __call__(self, doc):
        self.matcher.load()

        ents = list(doc.ents)
        for ent in ents:
            span_text = "##" + ent.text + "##"
            matches = self.matcher.ss_reader.retrieve(span_text)
            if not len(matches):
                continue
            best_match = max(
                (match for match in matches),
                key=lambda m: similarity(span_text, m, measure=self.matcher.measure),
            )
            cui = self.matcher.syn2cuis[best_match][0]
            ent.kb_id_ = cui
        doc.ents = ents

        return doc
