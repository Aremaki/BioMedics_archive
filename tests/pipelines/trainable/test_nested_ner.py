import spacy
from pytest import fixture
from spacy.tokens import Span
from spacy.training import Example

from edsnlp.utils.training import make_spacy_corpus_config, train


@fixture
def gold(blank_nlp):
    doc = blank_nlp.make_doc("Arret du ttt si folfox inefficace")

    # drug = "folfox"
    doc.ents = [Span(doc, 4, 5, "drug")]

    # event = "Arret"
    doc.spans["event"] = [Span(doc, 0, 1, "event")]

    # criteria = "si folfox inefficace"
    doc.spans["criteria"] = [Span(doc, 3, 6, "criteria")]

    return doc


def test_nested_ner_training(blank_nlp, gold):
    nlp = spacy.blank("eds")
    nlp.add_pipe("nested_ner")

    train(
        nlp,
        output_path="/tmp/test-train-edsnlp/",
        config=dict(
            **make_spacy_corpus_config(train_data=[gold], dev_data=[gold]),
            training=dict(max_steps=10, eval_frequency=5),
        ),
    )

    pred = nlp(gold.text)
    scores = nlp.pipeline[-1][1].score([Example(pred, gold)])

    assert scores["ents_f"] == 1.0
    assert len(pred.ents) == 1
    assert len(pred.spans["event"]) == 1
    assert len(pred.spans["criteria"]) == 1