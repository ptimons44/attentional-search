from gpt import callGPT
import spacy

import config
import json

import numpy as np

def get_k_most_similar_sents(content, query, k=20):
    """finds relevant content by compa

    Args:
        content (dict): keys are url (string), values are text scraped from the web
    """

    nlp = spacy.load(config.language_model())
    query_vec = nlp(query).vector

    most_similar_sents = []
    for source in content:
        doc = nlp(content[source])
        for sentence in doc.sents:
            simmilarity = sentence.vector.dot(query_vec) / (np.linalg.norm(sentence.vector)*np.linalg.norm(query_vec))
            most_similar_sents.append((source, sentence.text, simmilarity))
    
    most_similar_sents.sort(key=lambda x: x[2], reverse=True)
    return most_similar_sents[:min(k, len(most_similar_sents))]


if __name__ == "__main__":
    with open('/Users/patricktimons/Documents/GitHub/query-graph/query_graph/text.json', 'r') as f:
        content = json.load(f)
    query = "The 2020 election was stolen from Trump because of fraudulent voting machines and electronic ballots."
    top_k = get_k_most_similar_sents(content, query)
    for (source, sentence, simmilarity) in top_k:
        print(simmilarity, ":", sentence)