from query_graph import config

import numpy as np

import tiktoken
import openai
openai.api_key = config.OPENAI_APIKEY()


def callGPT(prompt, max_tokens=0):
    # if chatGPT 3.5 turbo
    if True:
      encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
      if len(encoding.encode(prompt)) >= 4090:
         return None
      response = openai.ChatCompletion.create(
         model=config.OPENAI_MODEL(),
         messages=[
            {"role": "user", "content": prompt}
         ]
      )
      return response.choices[0].message.content
    # if gpt 4
    else:
       pass


import numpy as np
import tiktoken
from openai.error import OpenAIError

# Given a list and a size, it will return that list in chunks of the given size
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

enc = tiktoken.encoding_for_model('text-embedding-ada-002')

def embed_sentences(sentences):
    MAX_SENTENCES = 2000  # Define the maximum number of sentences per batch
    
    all_embeddings = []
    
    for batch in chunks(sentences, MAX_SENTENCES):
        try:
            response = openai.Embedding.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            embeddings = [np.array(data["embedding"]) for data in response['data']]
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(len(batch))
            total = sum(len(enc.encode(sentence)) for sentence in batch)
            print("total", total)
            raise OpenAIError(f"OpenAI API Error: {str(e)}")
    
    return all_embeddings
