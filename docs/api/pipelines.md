# Pipelines

## Matcher

```{eval-rst}
.. automodule:: edsnlp.pipelines.generic.generic
```

## Pollution

```{eval-rst}
.. automodule:: edsnlp.pipelines.pollution.pollution
```

## Sentences

```{eval-rst}
.. automodule:: edsnlp.pipelines.sentences.sentences
```

## Dates

```{eval-rst}
.. automodule:: edsnlp.pipelines.dates.dates
```

## Sections

```{eval-rst}
.. automodule:: edsnlp.pipelines.sections.sections
```

## Normalizer

```{eval-rst}
.. automodule:: edsnlp.pipelines.normalizer.normalizer
```

## Negation

```{eval-rst}
.. automodule:: edsnlp.pipelines.negation.negation
```

## Family

```{eval-rst}
.. automodule:: edsnlp.pipelines.family.family
```

## Hypothesis

```{eval-rst}
.. automodule:: edsnlp.pipelines.hypothesis.hypothesis
```

## Antecedents

```{eval-rst}
.. automodule:: edsnlp.pipelines.antecedents.antecedents
```

## Reported Speech

```{eval-rst}
.. automodule:: edsnlp.pipelines.rspeech.rspeech
```

## QuickUMLS

```{eval-rst}
.. automodule:: edsnlp.pipelines.quickumls.quickumls
```

## Scores

### Base class

```{eval-rst}
.. automodule:: edsnlp.pipelines.scores.base_score
```

### Charlson Comorbidity Index

The `charlson` pipeline implements the above `score` pipeline with the following parameters:

```python
regex = [r"charlson"]

after_extract = r"charlson.*[\n\W]*(\d+)"

score_normalization_str = "score_normalization.charlson"

@spacy.registry.misc(score_normalization_str)
def score_normalization(extracted_score: Union[str,None]):
    """
    Charlson score normalization.
    If available, returns the integer value of the Charlson score.
    """
    score_range = list(range(0, 30))
    if (extracted_score is not None) and (int(extracted_score) in score_range):
        return int(extracted_score)
```