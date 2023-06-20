from gpt import callGPT
import spacy

import config
import json

import numpy as np

import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine

def sentence_embedding(sentence, model, tokenizer, context=""):
    # Add the special tokens.
    marked_text = "[CLS] " + sentence + " [SEP]"
    
    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    with torch.no_grad():

        outputs = model(tokens_tensor, segments_tensors)

        # Evaluating the model will return a different number of objects based on 
        # how it's  configured in the `from_pretrained` call earlier. In this case, 
        # becase we set `output_hidden_states = True`, the third item will be the 
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]

    # `hidden_states` has shape [13 x 1 x 22 x 768]

    # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = hidden_states[-2][0]

    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding

def similarity(sentence1, sentence2, model, tokenizer):
    enc1 = sentence_embedding(sentence1, model, tokenizer)
    enc2 = sentence_embedding(sentence2, model, tokenizer)
    return cosine(enc1, enc2)


def get_k_most_similar_sents(content, query, k=20):
    """finds relevant content by compa

    Args:
        content (dict): keys are url (string), values are text scraped from the web
    """
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased',
                                    output_hidden_states = True, # Whether the model returns all hidden-states.
                                    )
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    nlp = spacy.load(config.language_model())
    # query_vec = nlp(query).vector

    most_similar_sents = []
    for source in content:
        doc = nlp(content[source])
        for sentence in doc.sents:
            sim = similarity(sentence.text, query, model, tokenizer)
            # sim = sentence.vector.dot(query_vec) / (np.linalg.norm(sentence.vector)*np.linalg.norm(query_vec))
            most_similar_sents.append((source, sentence.text, sim))
    
    most_similar_sents.sort(key=lambda x: x[2])
    return most_similar_sents[:min(k, len(most_similar_sents))]


if __name__ == "__main__":
    with open('/Users/patricktimons/Documents/GitHub/query-graph/query_graph/text.json', 'r') as f:
        content = json.load(f)
    query = "The 2020 election was stolen from Trump because of fraudulent voting machines and electronic ballots."
    top_k = get_k_most_similar_sents(content, query)
    for (source, sentence, simmilarity) in top_k:
        print(simmilarity, ":", sentence)