from pathlib import Path

BASE_DIR = Path(__file__).parent

import spacy

from .evaluate import evaluate
from .pipelines.clean_entities import CleanEntities

if not spacy.tokens.Span.has_extension("event_type"):
    spacy.tokens.Span.set_extension("event_type", default=None)

for ext in ["Action","Certainty","Temporality","Negation","Family"]:
    if not spacy.tokens.Span.has_extension(ext):
        spacy.tokens.Span.set_extension(ext, default=None)

print("Monkey patching spacy.Language.evaluate")
spacy.Language.evaluate = evaluate