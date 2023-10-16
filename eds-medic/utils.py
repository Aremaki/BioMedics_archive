import pandas as pd 
import numpy as np
import spacy
from edsnlp.connectors.brat import BratConnector

def get_data(path_to_BRAT):
    brat = BratConnector(path_to_BRAT)
    nlp = spacy.blank("fr")
    df = brat.brat2docs(nlp,  run_pipe=False)
    return df


def get_drug_of_doc(doc):
    c = 0
    ent_name = []
    for ent in doc.ents:
        if ent.label_ == "Chemical_and_drugs":
            c+=1
            ent_name.append(ent.text)
    return [c,ent_name]


def compute_score(gt,pred, debug=False):
    
    gt = [token.text for token in gt]
    pred = [token.text for token in pred]
    
    gt = list(set(gt))
    pred = list(set(pred))
    
    FN = 0
    TP = 0
    FP = 0
    
    for i in range(len(pred)):
        if pred[i] in gt:
            TP+=1
        else:
            FP +=1
    for j in range(len(gt)):
        if gt[j] in pred:
            pass
        else:
            FN+=1
    
    if debug:
        print('TP',TP, 'FP', FP, 'FN', FN)
    try:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * (precision * recall) / (precision + recall)
    except:
        precision = -1
        recall = -1
        F1 = -1
    return precision, recall, F1
