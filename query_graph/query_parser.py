import json

import pandas as pd
import numpy as np

import spacy
import gpt

def extract_keywords(input, nlp):
   """_summary_

   Args:
       input (string): query from which we want to extract keywords
       nlp (space.Language): a model (either off-the-shelf or custom) that creates Doc object

   Returns:
       list: list of strings, where each string is a noun phrase
   """
   doc = nlp(input)
   return list(doc.noun_chunks)

def augment_keywords(keywords):
   similar = set()
   prompt = "You are part of the pipeline in a research assistant that generates a knowledge graph given an input query question. \
      Given the following list of noun phrases to each be searched in a search enging, augment the list with phrases that will \
         enhance the search process. List of current keywords: " + str(keywords) + ". Output your resonse as a json where the key of the \
            augmented list is \"keywords\". For instance, if your list of augmented keywords is [\"The 2020 election\", \"Trump\", \"fraudulent \
               voting machines\", \"electronic ballots\"], then your response should be \"{\"keywords\": [\"The 2020 election\", \"Trump\", \
                  \"fraudulent voting machines\", \"electronic ballots\"]}\"."
   response = json.loads(gpt.callGPT(prompt))
   return response["keywords"]

# WIP
def generate_search_queries(query):
    keywords = extract_keywords(query)
    augment_keywords = augment_keywords(keywords)
    return augment_keywords
   #  search_queries = set()
   #  # enumerate all possible combinations of keywords with AND
   #  for keywords_per_combo in range(len(keywords)):
   #     search_queries.add()


if __name__ == "__main__":

   nlp = spacy.load("en_core_web_sm") # set with config file / import custom model
   input_string = "The 2020 election was stolen from Trump because of fraudulent voting machines and electronic ballots."
   keywords = extract_keywords(input_string, nlp)
   print(keywords)
   # print(augment_keywords(keywords))
