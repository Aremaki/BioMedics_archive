import spacy
from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.matchers.regex import RegexMatcher
from edsnlp.pipelines.misc.measurements.factory import DEFAULT_CONFIG
from edsnlp.pipelines.misc.measurements.measurements import (
    MeasurementsMatcher,
    RangeMeasurement,
    SimpleMeasurement,
)
from edsnlp.pipelines.misc.measurements.patterns import units_config
from edsnlp.utils.filter import filter_spans


def augment_time_of_day(terms):
    return [
        aug
        for term in terms
        for aug in (
            f"le {term}",
            f"par {term}",
            f"/ {term}",
            f"tous les {term}",
            f"toutes les {term}",
            f"chaque {term}",
            f"au {term}",
            f"après le {term}",
            f"après les {term}",
            f"pendant le {term}",
            f"pendant les {term}",
            f"avant le {term}",
            f"avant les {term}",
            f"à {term}",
            term,
        )
    ]


class FrequencyMatcher(object):
    def __init__(self, nlp, threshold=3, merge_mode: str = "union"):
        self.nlp = nlp
        self.threshold = threshold
        divisors = [
            "par",
            "fois par",
            "fois /",
            "tous les",
            "ttes les",
            "tte les",
            "toutes les",
            "toute les",
            "x par",
            "x /",
        ]
        self.measurement_matcher = MeasurementsMatcher(
            nlp,
            **{
                **DEFAULT_CONFIG,
                **{
                    "compose_units": False,
                    "units_config": {
                        **{k: v for k, v in units_config.items() if v["dim"] == "time"},
                        "day": {
                            "dim": "time",
                            "degree": 1,
                            "scale": 3600 * 24,
                            "terms": [
                                "jour",
                                "jours",
                                "j",
                                "jour tous les jours",
                                "j tous les jours",
                                "jours tous les jours",
                            ],
                            "followed_by": None,
                        },
                        "week": {
                            "dim": "time",
                            "degree": 1,
                            "scale": 3600 * 24 * 7,
                            "terms": [
                                "sem",
                                "semaine",
                                "semaines",
                            ],
                            "followed_by": None,
                        },
                        "two-day": {
                            "dim": "time",
                            "degree": 1,
                            "scale": 3600 * 24 * 2,
                            "terms": [
                                "jour pair",
                                "jours pairs",
                                "jour tous les jours pairs",
                                "j tous les jours pairs",
                                "jours tous les jours pairs",
                                "jour impair",
                                "jours impairs",
                                "jour tous les jours impairs",
                                "j tous les jours impairs",
                                "jours tous les jours impairs",
                            ],
                            "followed_by": None,
                        },
                        "per_morning": {
                            "dim": "time",
                            "degree": -1,
                            "scale": 1 / (3600 * 24),
                            "terms": augment_time_of_day(
                                [
                                    "matin",
                                    "petit-déjeuner",
                                    "petit déjeuner",
                                    "matins",
                                    "petit-déjeuners",
                                    "petit déjeuners",
                                ]
                            ),
                            "followed_by": None,
                        },
                        "per_evening": {
                            "dim": "time",
                            "degree": -1,
                            "scale": 1 / (3600 * 24),
                            "terms": augment_time_of_day(
                                [
                                    "diner",
                                    "coucher",
                                    "soir",
                                    "nuit",
                                    "diners",
                                    "couchers",
                                    "soirs",
                                    "nuits",
                                ]
                            ),
                            "followed_by": None,
                        },
                        "per_lunch": {
                            "dim": "time",
                            "degree": -1,
                            "scale": 1 / (3600 * 24),
                            "terms": augment_time_of_day(
                                ["déjeuner", "midi", "déjeuners", "midis", "après-midi"]
                            ),
                            "followed_by": None,
                        },
                        "per_meal": {
                            "dim": "time",
                            "degree": -1,
                            "scale": 3600 * 24 / 3,
                            "terms": augment_time_of_day(
                                ["repas", "collation", "collations"]
                            ),
                            "followed_by": None,
                        },
                        "per_day": {
                            "dim": "time",
                            "degree": -1,
                            "scale": 1 / (3600 * 24),
                            "terms": [
                                "quotidien",
                                "quotidiens",
                                "journalier",
                                "journaliers",
                                "quotidienne",
                                "quotidiennes",
                                "journaliere",
                                "journalieres",
                            ],
                            "followed_by": None,
                        },
                        "per_week": {
                            "dim": "time",
                            "degree": -1,
                            "scale": 1 / (3600 * 24 * 7),
                            "terms": [
                                "hebdomadaire",
                                "hebdomadaires",
                                "jours par semaine",
                                "jour par semaine",
                            ],
                            "followed_by": None,
                        },
                        "per_month": {
                            "dim": "time",
                            "degree": -1,
                            "scale": 1 / (3600 * 24 * 7 * 30.4167),
                            "terms": ["mensuel", "mensuelle", "mensuels", "mensuelles"],
                            "followed_by": None,
                        },
                    },
                    "unit_divisors": ["/", *divisors],
                    "stopwords": ("par", "sur", "de", "a", ":", "-"),
                    "merge_mode": merge_mode,
                    "measurements": {
                        "time": {"unit": "day"},
                        "per_time": {"unit": "per_day"},
                    },
                },
            },
        )
        self.term_matcher = EDSPhraseMatcher(nlp.vocab, "NORM")
        self.term_matcher.build_patterns(
            nlp,
            {
                "unit_divisor": divisors,
                "measure_divisor": ["sur"],
                "divisor": ["/"],
            },
        )
        self.regex_matcher = RegexMatcher(attr="LOWER")
        # fmt: off
        self.regex_matcher.build_patterns({
            "day_of_month": [r"\bJ[1-9][0-9]?(?:-J[1-9][0-9]?)?\b"],
            "time_of_day": [
                r"(?<![-/0-9])\b(?P<morning>[1-9]?[0-9])[ ]?-[ ]?(?P<lunch>[1-9]?[0-9])[ ]?-[ ]?(?P<evening>[1-9]?[0-9])\b(?![-/0-9])",  # noqa: E501
                r"(?<![./0-9])\b(?P<morning>[1-9]?[0-9])[ ]?[.][ ]?(?P<lunch>[1-9]?[0-9])[ ]?[.][ ]?(?P<evening>[1-9]?[0-9])\b(?![./0-9])",  # noqa: E501
                r"(?<![/0-9][ ]*)\b(?P<morning>[1-9]?[0-9]) (?P<lunch>[1-9]?[0-9]) (?P<evening>[1-9]?[0-9])\b(?![ ]*[/0-9])",  # noqa: E501
                r"(?<!\bce\s*)\b(?:le\s*)?(?:(?P<morning>matin)|(?P<lunch>midi)|(?P<evening>soir|coucher|couché|nuit))\b",  # noqa: E501
                r"\b(?:(?:avant|a|à|apres|pendant|au)\s*)?(?:(?:les|le|chaque)\s*)?(?:(?P<meal>repas|collations?)|(?P<evening>diner|couché|coucher)s?|(?P<noon>d[eé]jeuners?)|(?P<morning>petits?[- ]?d[eé]jeuners?))\b",  # noqa: E501
            ],
            "day_of_week": [
                r"\b(?:le|les|chaque\s*)?(?:lundi|mardi|mercredi|jeudi|vendredi|samedi)\b",  # noqa: E501
            ],
        })
        # fmt: on

    def __call__(self, doc):

        existing = [
            ent
            for ent in (*doc.ents, *doc.spans.get("frequencies", ()))
            if ent.label_ == "frequency"
        ]
        other_ents = [ent for ent in doc.ents if ent.label_ != "frequency"]
        doc.ents = other_ents

        unit_registry = self.measurement_matcher.unit_registry
        time_dim_str = unit_registry.parse_unit("day")[0]
        per_time_dim_str = unit_registry.parse_unit("per_day")[0]

        # Special regex patterns
        special_frequencies = []
        last = None
        stopwords = {"et", "ou", ",", "/"}
        for span, groupdict in filter_spans(
            self.regex_matcher(doc, as_spans=True, return_groupdict=True)
        ):
            if span.label_ == "time_of_day":
                values = []
                for key, value in groupdict.items():
                    if value:
                        try:
                            value = int(value)
                        except (ValueError, TypeError, SyntaxError):
                            value = 3 if key == "meal" else 1
                        values.append(value)
                span._.value = SimpleMeasurement(sum(values), "per_day", unit_registry)
            elif span.label_ == "day_of_week":
                span._.value = SimpleMeasurement(1, "per_week", unit_registry)

            # If we can merge with the previous frequency
            # TODO: refactor this to have a consistent handling of stopwords
            if last is not None:
                try:
                    inbetween_tokens = set(
                        token.text for token in doc[last.end : span.start]
                    )
                    if last.label == span.label and inbetween_tokens < stopwords:
                        merged_value = span._.value + last._.value
                        special_frequencies[-1] = spacy.tokens.Span(
                            doc, last.start, span.end, span.label_
                        )
                        special_frequencies[-1]._.value = merged_value
                        last = special_frequencies[-1]
                        continue
                except AttributeError:
                    pass
            # otherwise
            special_frequencies.append(span)
            last = span

        for span in special_frequencies:
            span.label_ = "frequency"

        # Measurement-based frequencies
        divisors = self.term_matcher(doc, as_spans=True)
        measurements, unmatched = self.measurement_matcher.extract_measurements(doc)

        measurements = self.measurement_matcher.extract_ranges(measurements)

        freq_matches = []
        time_matches = []

        numbers = []
        for unit in unmatched:
            if unit.label in self.measurement_matcher.number_label_hashes:
                numbers.append(unit)
                continue
            try:
                dims = unit_registry.parse_unit(unit.label_)[0]
            except (KeyError, AttributeError, TypeError):
                continue
            if dims == time_dim_str:
                unit._.value = SimpleMeasurement(1, unit.label_, unit_registry)
                time_matches.append(unit)
            if dims == per_time_dim_str:
                unit._.value = SimpleMeasurement(1, unit.label_, unit_registry)
                unit.label_ = "frequency"
                freq_matches.append(unit)
        for measurement in measurements:
            if unit_registry.parse_unit(measurement._.value.unit)[0] == time_dim_str:
                time_matches.append(measurement)
            else:
                measurement.label_ = "frequency"
                freq_matches.append(measurement)

        freq_matches = [
            m
            for m in freq_matches
            if all(v.per_hour <= self.threshold for v in m._.value)
        ]
        time_matches = [
            m
            for m in time_matches
            if all(v.hour >= 1 / self.threshold for v in m._.value)
        ]

        matches = filter_spans((*time_matches, *numbers, *divisors))
        i = 0
        while i < len(matches) - 1:
            match = matches[i]
            next_match = matches[i + 1]
            if (
                (match.label_ == "divisor" or match.label_ == "unit_divisor")
                and next_match in measurements
                and doc[match.end : next_match.start].text.strip() == ""
            ):
                new_match = spacy.tokens.Span(
                    doc, match.start, next_match.end, label="frequency"
                )
                value = next_match._.value
                if isinstance(value, SimpleMeasurement):
                    new_match._.value = SimpleMeasurement(
                        1 / value.value, f"per_{value.unit}", unit_registry
                    )
                else:
                    new_match._.value = RangeMeasurement(
                        1 / value.value[1],
                        1 / value.value[0],
                        f"per_{value.unit}",
                        unit_registry,
                    )
                freq_matches.append(new_match)
                i += 2
                continue
            elif (
                i < len(matches) - 2
                and match in measurements
                and (
                    next_match.label_ == "measure_divisor"
                    or next_match.label_ == "divisor"
                )
                and matches[i + 2] in numbers
                and doc[match.end : next_match.start].text.strip() == ""
                and doc[next_match.end : matches[i + 2].start].text.strip() == ""
            ):
                value = match._.value
                div = matches[i + 2]
                if div.label_ == "number":
                    number_text = (
                        div.text.replace(" ", "")
                        .replace(",", ".")
                        .replace(" ", "")
                        .lstrip("0")
                    )
                    divide_value = eval(number_text or "0")
                else:
                    divide_value = float(div.label_)
                new_match = spacy.tokens.Span(
                    doc, match.start, div.end, label="frequency"
                )
                if isinstance(value, SimpleMeasurement):
                    new_match._.value = SimpleMeasurement(
                        value.value / divide_value, f"per_{value.unit}", unit_registry
                    )
                else:
                    new_match._.value = RangeMeasurement(
                        value.value[1] / divide_value,
                        value.value[0] / divide_value,
                        f"per_{value.unit}",
                        unit_registry,
                    )

                freq_matches.append(new_match)
                i += 3
                continue
            else:
                i += 1

        final_matches = self.measurement_matcher.merge_with_existing(
            [*special_frequencies, *freq_matches],
            existing,
        )

        doc.spans["frequencies"] = filter_spans(final_matches)
        doc.ents = filter_spans((*other_ents, *final_matches))

        return doc
