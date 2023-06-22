from information_extraction.gpt import callGPT

import config
import json

import numpy as np

import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine

import re
# -*- coding: utf-8 -*-
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'

def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences


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

def get_k_most_similar_sents(content, query, k=20, context_window=2):
    """finds relevant content by compa

    Args:
        content (dict): keys are url (string), values are tuple: (search_phrase, text scraped from the web)
    """
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased',
                                    output_hidden_states = True, # Whether the model returns all hidden-states.
                                    )
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    most_similar_sents = []
    for source in content:
        (search_phrase, page) = content[source]
        sentences = split_into_sentences(page)
        for ix, sentence in enumerate(sentences):
            pre_context = ""
            post_context = ""
            if ix > 0 and context_window > 0:
                pre_context = " ".join(sentences[max(0, ix-context_window):ix]).strip()
            if ix < len(sentences) - 1 and context_window > 0:
                post_context = " ".join(sentences[ix+1:min(len(sentences)-1, ix+context_window+1)]).strip()
            context = pre_context + " " + sentence + " " + post_context
            # get similarity
            sim = similarity(sentence, query, model, tokenizer)
            # add to list
            most_similar_sents.append((sim, source, sentence, context, search_phrase))
    
    most_similar_sents.sort(key=lambda x: x[0])
    return most_similar_sents[:min(k, len(most_similar_sents))]


if __name__ == "__main__":
    with open('/Users/patricktimons/Documents/GitHub/query-graph/temp.json', 'r') as f:
        content = json.load(f)
    # query = "The 2020 election was stolen from Trump because of fraudulent voting machines and electronic ballots."
    query = "Is climate change a timely threat to humanity?"
    top_k = get_k_most_similar_sents(content, query)
    for ix, (sim, source, sentence, context) in enumerate(top_k):
        print(ix,":", sentence, ":\n", context, "\n\n")