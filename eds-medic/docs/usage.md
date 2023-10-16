# Usage

## Loading the model

To use the default model, you can simply run:

```python
import eds_medic

nlp = eds_medic.load()
nlp("Le patient prend du paracétamol 500 mg 3 fois par jour pendant 5 jours.")

# What are the components of this pipeline ?
print(nlp.pipe_names)
# ['tok2vec', 'ner', 'qualifier']
```

The pipeline is not yet complete: it only contains:

- an embedding layer (`tok2vec`)
- a named entity recognizer (`ner`)
- and a span classification component (`qualifier`).

You will need to add

- the [`eds.dates` component](https://aphp.github.io/edsnlp/latest/pipelines/misc/dates) to extract dates
- the `doses` to normalize the extracted drug doses
- the `frequencies` component to normalize the extracted frequencies
- the `drugs_normalizer` component to normalize the extracted drugs to their ATC codes

```python
import eds_medic

nlp = eds_medic.load()
# Extract sentences
nlp.add_pipe("eds.sentences", before="qualifier")
# Update token attributes such as NORM or TAG, detect pollution, etc
nlp.add_pipe("eds.normalizer", before="qualifier")
# Extract dates
nlp.add_pipe(
    "eds.dates",
    before="qualifier",
    config={
        "use_date_label": True,
        "as_ents": True,
    },
)
# Normalize frequencies extracted by the NER component
nlp.add_pipe(
    "frequencies",
    before="qualifier",
    config={
        "merge_mode": "intersect",
    },
)
# Normalize doses extracted by the NER component
nlp.add_pipe(
    "doses",
    before="qualifier",
    config={
        "merge_mode": "intersect",
    },
)
# Normalize drugs extracted by the NER component
nlp.add_pipe("drugs_normalizer")
```

## Extracting medication

```python
# ↑ Omitted code above ↑

doc = nlp("Le patient prend du paracétamol 500 mg 3 fois par jour pendant 5 jours.")

# Table formatting
print("{:<20} | {:<15} | {:<15} | {:<15}".format("text", "type", "event", "normalized"))
print("{:-<20} | {:-<15} | {:-<15} | {:-<15}".format("", "", "", ""))

for ent in doc.ents:
    if ent._.value:
        normalized = ent._.value
    elif ent._.date:
        normalized = ent._.date.norm()
    else:
        normalized = ent.kb_id_ or ""
    print(
        "{:<20} | {:<15} | {:<15} | {:<15}".format(
            str(ent),
            str(ent.label_),
            str(ent._.event_type or ""),
            str(normalized),
        )
    )
```

| text            | type      | event | normalized    |
|-----------------|-----------|-------|---------------|
| paracétamol     | drug      | start | N02BE01       |
| 500 mg          | dose      |       | 500 mg        |
| 3 fois par jour | frequency |       | 3 per_day     |
| pendant 5 jours | duration  |       | during 5 days |
