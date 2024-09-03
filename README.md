<a href="https://aphp.github.io/eds-medic/" target="_blank">
    <img src="https://img.shields.io/badge/docs-passed-brightgreen" alt="Documentation">
</a>
<a href="https://github.com/psf/black" target="_blank">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
</a>
<a href="https://python-poetry.org" target="_blank">
    <img src="https://img.shields.io/badge/reproducibility-poetry-blue" alt="Poetry">
</a>
<a href="https://dvc.org" target="_blank">
    <img src="https://img.shields.io/badge/reproducibility-dvc-blue" alt="DVC">
</a>

# EDS-Medic

EDS-Medic is a project used at APHP to extract drug prescriptions from clinical reports.
It is built on top of [edsnlp](https://github.com/aphp/edsnlp) and [spaCy](https://spacy.io/).

## Installation

```bash
pip install eds-medic

# To use with a GPU, you will need to specify your version of CUDA
pip install 'eds-medic[cuda110]'
```

## Loading the model

```python
import eds_medic

nlp = eds_medic.load()
# add components to the model and/or run it on documents
```

## Training a model

We created a single workflow that:

- Converts the datasets from JSONL or BRAT files to spaCy format
- Trains the pipeline
- Evaluates the pipeline using the test set
- Packages the resulting model to make it pip-installable

To use it, you will need to supply:

- A labelled dataset
- A HuggingFace transformers model, or use `camembert-base`

In any case, you will need to modify the configuration to reflect these changes.

## Entities and attributes

The default model is trained to recognize the following entities:

| Entity      | Description                        |
|-------------|------------------------------------|
| `drug`      | the drug name                      |
| `class`     | a active substance class           |
| `dose`      | the dose of the drug               |
| `frequency` | a frequency for the drug intake    |
| `criteria`  | a criteria for the drug intake     |
| `duration`  | how long the prescription is valid |
| `route`     | the route of the drug intake       |

And extract the following attributes:

| Attribute          | Description    | Entities                |
|--------------------|----------------|-------------------------|
| `ent._.event_type` | the event type | `drug`, `class`, `date` |

## Commands

| Command           | Description                                                |
|-------------------|------------------------------------------------------------|
| `convert`         | Convert data to spaCy's binary format                      |
| `train`           | Train the NER model                                        |
| `evaluate`        | Evaluate the model and export metrics                      |
| `package`         | Package the trained model as a pip package                 |
| `visualize-model` | Visualize the model's output interactively using Streamlit |

Run the command with
```bash
spacy project run [command] [options]
```

## Documentation

Visit the [documentation](https://datasciencetools-pages.eds.aphp.fr/eds-medic) for more information!

## Acknowledgement

We would like to thank [Assistance Publique – Hôpitaux de Paris](https://www.aphp.fr/)
and [AP-HP Foundation](https://fondationrechercheaphp.fr/) for funding this project.
