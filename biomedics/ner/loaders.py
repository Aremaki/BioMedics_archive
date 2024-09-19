import eds_biomedic_aphp  # type: ignore


def eds_biomedic():
    """
    EDS-Biomedic model loaded in the edsnlp fashion. EDS-Biomedic is a model that is
    able to retrieve biological measurements as well as drug uses and other information
    within textual reports.
    Args:
    Returns:
        The described model.
    """
    nlp = eds_biomedic_aphp.load()

    return nlp
