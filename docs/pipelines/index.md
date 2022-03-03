# Pipelines overview

EDS-NLP's main goal is to provide easy-to-use SpaCy pipelines.

=== "Core"

    | Pipeline               | Description                                     |
    | ---------------------- | ----------------------------------------------- |
    | `eds.normalizer`       | Non-destructive input text normalization        |
    | `eds.sentences`        | Better sentence boundary detection              |
    | `eds.matcher`          | A simple yet powerful entity extractor          |
    | `eds.advanced-matcher` | A conditional entity extractor                  |
    | `eds.endlines`         | An unsupervised model to classify each end line |

=== "Qualifiers"

    | Pipeline               | Description                                     |
    | ---------------------- | ----------------------------------------------- |
    | `eds.negation`         | Rule-based negation detection                   |
    | `eds.family`           | Rule-based family context detection             |
    | `eds.hypothesis`       | Rule-based speculation detection                |
    | `eds.reported_speech`  | Rule-based reported speech detection            |
    | `eds.antecedent`       | Rule-based antecedent detection                 |

=== "Miscellaneous"

    | Pipeline               | Description                                     |
    | ---------------------- | ----------------------------------------------- |
    | `eds.dates`            | Date extraction and normalization               |
    | `eds.sections`         | Section detection                               |
    | `eds.reason`           | Rule-based hospitalisation reason detection     |

=== "NER"

    | Pipeline                 | Description                |
    | ------------------------ | -------------------------- |
    | `eds.charlson`           | A Charlson score extractor |
    | `eds.sofa`               | A SOFA score extractor     |
    | `eds.emergency.priority` | A priority score extractor |
    | `eds.emergency.ccmu`     | A CCMU score extractor     |
    | `eds.emergency.gemsa`    | A GEMSA score extractor    |