# Extraction

## Annotated Entities

We annotated clinical documents with the following entities :

- `drug` : the drug name
- `class` : an active substance class
- `dose` : the dose of the drug
- `frequency` : a frequency for the drug intake
- `criteria` : a criteria for the drug intake
- `duration` : how long the prescription is valid
- `route` : the route of the drug intake

Additionally, we annotated event-entities on event triggers and dates to be able to
classify prescriptions starts, stops and changes. These events have been assigned
to `drug`, `class` and `date` entities under the:

- `drug._.event_type` attribute that can take the following values

    * `start`: start of a prescription
    * `stop`: end of a prescription
    * `start`-stop: unique drug intake or on a limited period
    * `increase`: increase of the dose / frequency
    * `decrease`: decrease of the dose / frequency
    * `switch`: switch to another drug

We also annotated `blobs` to identify the prescription text and detect relation between the entities, but no model is currently trained on these annotations.

## Data Selection

TBC

## Software

The software used to annotate the document with personal identification entities was
[Metanno](https://github.com/percevalw/metanno/), but any software will do.

The `convert` step takes as input either a jsonlines file (`.jsonl`) or a folder
containing Standoff files (`.ann`) from an annotation with [Brat](https://brat.nlplab.org/).

Feel free to [submit a pull request](https://github.com/aphp/eds-medic/pulls) if these
formats do not suit you!
