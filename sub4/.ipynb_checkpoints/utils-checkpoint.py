import torch
import nltk
import numpy as np

def img2feature(object_visual, feature_extractor):
    image_features = feature_extractor(images=object_visual, return_tensors="pt")['pixel_values']
    
    return image_features

def normalize_sentence(sentence):
    """Normalize the sentences and tokenize."""
    return nltk.tokenize.word_tokenize(sentence.lower())

def CalBELU(list_predicted, list_target):
    """
    list_predicted = ["hello my name is joosung", ..., ]
    list_target = ["hi my name is joo", ..., ]
    """
    
    # Compute BLEU scores.
    bleu_scores = []
    # Smoothing function.
    chencherry = nltk.translate.bleu_score.SmoothingFunction()

    for response, gt_response in zip(list_predicted, list_target):
        bleu_score = nltk.translate.bleu_score.sentence_bleu(
            [normalize_sentence(gt_response)],
            normalize_sentence(response),
            smoothing_function=chencherry.method7,
        )
        bleu_scores.append(bleu_score)
    
    return np.mean(bleu_scores), np.std(bleu_scores) / np.sqrt(len(bleu_scores))