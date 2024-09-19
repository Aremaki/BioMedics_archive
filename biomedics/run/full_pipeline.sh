#!/bin/bash

# File for the full pipeline of the EDS Biomedic pipeline which is:
# - apply NER to documents 
# TODO: retake function in aiT that make corpus from brat files or from a pandas df
# TODO: test the function that convert ents and spans to df
# - apply the extraction of measurements
# TODO: test the function that clean the measurements extracted
# TODO: change to work with a pandas dataframe instead of a BRAT folder
# - apply the normalization of labels
# - make a yaml file with sections for each part of the pipeline
