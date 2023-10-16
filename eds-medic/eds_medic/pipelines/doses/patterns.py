from edsnlp.pipelines.misc.measurements.patterns import (  # noqa: F401
    number_terms,
    stopwords,
    unit_divisors,
    units_config,
)

_default_terms = {t for config in units_config.values() for t in config["terms"]}

units_config = {
    **{name: cfg for name, cfg in units_config.items() if not cfg["dim"] == "time"},
    "pill": {
        "dim": "pill",
        "degree": 0,
        "scale": 1,
        "terms": [
            "cpm",
            "comprimé",
            "comprimés",
            "cp",
            "cpr",
            "sachet",
            "sachets",
            "tube",
            "tubes",
            "dose",
            "doses",
            "unité",
            "unités",
            "ampoule",
            "ampoules",
            "injection",
            "injections",
            "mesure",
            "mesures",
            "gélule",
            "gélules",
            "bouffée",
            "bouffées",
            "prise",
            "prises",
        ],
        "followed_by": None,
    },
}

units_config = {
    name: {
        **config,
        "terms": config["terms"]
        + [t + "x" for t in config["terms"] if t + "x" not in _default_terms],
    }
    for name, config in units_config.items()
}

measurements = [
    {
        "name": "dose",
        "unit": "kg",
    },
    {
        "name": "dose",
        "unit": "pill",
    },
    {
        "name": "dose",
        "unit": "m3",
    },
    {
        "name": "dose",
        "unit": "ui",
    },
    {
        "name": "dose",
        "unit": "g_per_m2",
    },
    {
        "name": "dose",
        "unit": "g_per_m3",
    },
]
