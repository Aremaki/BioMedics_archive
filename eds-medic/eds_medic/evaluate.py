from typing import Optional, Any, Dict, Iterable

from spacy.language import _copy_examples
from spacy.training import validate_examples, Example
from timeit import default_timer as timer


def evaluate(
    self,
    examples: Iterable[Example],
    *,
    batch_size: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Evaluate a model's pipeline components.

    Parameters
    ----------
    examples : Iterable[Example]
        `Example` objects.
    batch_size : Optional[int]
        Batch size to use.

    Returns
    -------
    Dict[str, Any]
        The evaluation results.
    """
    examples = list(examples)
    validate_examples(examples, "Language.evaluate")
    examples = _copy_examples(examples)
    if batch_size is None:
        batch_size = self.batch_size

    scores = {}

    total_time = 0

    begin_time = timer()
    # this is purely for timing
    for eg in examples:
        self.make_doc(eg.reference.text)
    total_time += timer() - begin_time

    n_words = sum(len(eg.predicted) for eg in examples)

    predictions = [eg.predicted for eg in examples]

    for name, component in self.pipeline:
        begin_time = timer()
        docs = [doc.copy() for doc in predictions]
        docs = list(component.pipe(docs, batch_size=batch_size))
        total_time += timer() - begin_time

        if name == "tok2vec":
            predictions = docs

        if hasattr(component, "score"):
            scores.update(
                component.score(
                    [Example(doc, eg.reference) for doc, eg in zip(docs, examples)]
                )
            )

    scores["speed"] = n_words / total_time
    return scores
