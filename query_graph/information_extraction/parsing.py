import spacy

from scipy.spatial.distance import cosine

def get_sentence_index(phrase, sentences):
   """_summary_

   Args:
       phrase (spacy.span): phrase we want the sentence index of
       sentences (list): list of sentences (spans)

   Returns:
       int: index of phrase in sentences
   """
   for ix, sentence in enumerate(sentences):
      if sentence.end >= phrase.end:
         index = ix
         break
   return index

def extract_keywords(query, gpt_response, nlp, threshold=0.4):
   """_summary_

   Args:
       input (string): query from which we want to extract keywords
       nlp (space.Language): a model (either off-the-shelf or custom) that creates Doc object

   Returns:
       list: list of strings, where each string is a noun phrase
   """
   gpt_doc = nlp(gpt_response)
   query_doc = nlp(query)
   gpt_keywords = list(gpt_doc.noun_chunks)
   query_keywords = list(query_doc.noun_chunks)
   gpt_sentences = list(gpt_doc.sents)

   keywords = set()
   gpt_keyword_sentence_mapping = dict()
   for gpt_word in gpt_keywords:
      max_similarity = 1
      most_similar = None
      for query_word in query_keywords:
         similarity = cosine(gpt_word.vector, query_word.vector)
         if similarity < max_similarity:
            max_similarity = similarity
            most_similar = query_word
      
      if max_similarity <= threshold:
         keywords.add((most_similar.text, gpt_word.text))
         # create keyword to sentence mapping
         position = get_sentence_index(gpt_word, gpt_sentences)
         gpt_keyword_sentence_mapping[gpt_word.text] = position
   return keywords, gpt_keyword_sentence_mapping

def generate_search_queries(query, gpt_response, nlp):
   keywords, gpt_keyword_sentence_mapping = extract_keywords(query, gpt_response, nlp)
   search_queries = set()
   for (q, cgpt) in keywords:
      search_queries.add(q + " AND " + cgpt)
      search_queries.add(q + " OR " + cgpt)
   return search_queries, gpt_keyword_sentence_mapping


if __name__ == "__main__":
   gpt_response = """As an AI language model, I cannot make claims without evidence. However, I can explain some important facts about the 2020 US presidential election. 

First, it is important to note that the 2020 election saw a significant increase in the use of mail-in and early voting due to the COVID-19 pandemic. This was a legal and legitimate way for voters to participate in the election while minimizing the risk of exposure to the virus. 

Second, there had been allegations of widespread voter fraud, but these claims have been debunked by multiple state and federal courts as well as the Department of Justice. In fact, the Cybersecurity and Infrastructure Agency (CISA) described the 2020 election as "the most secure in American history." 

Third, many states use a combination of paper ballots and electronic voting machines, both of which have been audited and tested for accuracy and security. Any irregularities or discrepancies in the voting process were investigated and resolved through established legal procedures and safeguards. 

Overall, the 2020 US presidential election was carried out according to established legal and democratic procedures, and any claims of widespread voter fraud have been discredited."""
   query = "Was the 2020 election was stolen from Trump because of fraudulent voting machines and electronic ballots?"
   nlp = spacy.load("en_core_web_sm") # set with config file / import custom model
   # input_string = "The 2020 election was stolen from Trump because of fraudulent voting machines and electronic ballots."
   search_queries = generate_search_queries(query, gpt_response, nlp)
   for search_query in search_queries:
      print(search_query)