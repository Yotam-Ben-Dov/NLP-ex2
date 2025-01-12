# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:27:40 2024

@author: yotam
"""

from transformers import RobertaTokenizer, RobertaModel, pipeline
import torch
import numpy as np


SIMILAR = ["i can take a right", "you should turn right"]
NOTSIMILAR = ["Thou shall carry me like a princess", "princesses are royalty"]

def main():
    # warmup
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base")
    predictions = so_mask(tokenizer, model)
    similar = similarity(tokenizer, model, SIMILAR)
    not_similar = similarity(tokenizer, model, NOTSIMILAR)
    print(predictions, similar, not_similar)
    
def so_mask(tokenizer, model):
    # 1.1
    text = "I am so <mask>"
    encoded_input = tokenizer(text, return_tensors="pt")
    outputs = model(**encoded_input, output_hidden_states=True)
    hidden_states = outputs.last_hidden_state
    # the fectors for am and <mask>
    am_vector = hidden_states[0][2]
    mask_vector = hidden_states[0][4]
    # 1.2
    # return the predictions for mask (am is 1.0 for 'am' and 0 for every other suggestion)
    unmasker = pipeline("fill-mask", model='roberta-base')
    return unmasker(text, top_k=5)

def similarity(tokenizer, model, sentences):
    # 2 and 3 
    # Encode sentences
    encoded = encode_sentences(tokenizer, model, sentences)
    vecs = [s[0][1].detach().numpy() for s in encoded]
    # Calculate cosine similarities
    return cosine_similarity(vecs[0], vecs[1])

def encode_sentences(tokenizer, model, sentences):
    encoded = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt')
        outputs = model(**inputs)
        encoded.append(outputs.last_hidden_state)
    return encoded

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

if __name__ == "__main__":
    main()