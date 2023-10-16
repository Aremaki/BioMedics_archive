# Quickstart

## Deployment

This project trains our prescription extraction pipeline, and make it pip-installable.

## Requirements

To use this repository, you will need to supply:

- A labelled dataset
- A HuggingFace transformers model, or use a publicly available model like `camembert-base`

In any case, you will need to modify the configuration to reflect these changes.

## Installation

Install the requirements by running the following command at the root of the repo

```bash
poetry install
```

## Training a model

EDS-Medic is a [spaCy project](https://spacy.io/usage/projects).
We created a single workflow that:

- Converts the datasets to spaCy format
- Trains the pipeline
- Evaluates the pipeline using the test set
- Packages the resulting model to make it pip-installable

To add a new dataset, run

```bash
dvc import-url url/or/path/to/your/dataset data/dataset
```

To (re-)train a model and package it, just run:

```bash
dvc repro
```

You should now be able to install and publish it:

```bash
pip install dist/eds_medic-[VERSION NUMBER]-*
```

## Use it

To use it, execute

```python
import eds_medic

nlp = eds_medic.load()
doc = nlp(
    """On prescrit du doliprane 200mg au patient, à prendre 2 / jour pendant 3 mois"""
)
for ent in doc.ents:
    print(ent, ent.label)

# TBC
```
