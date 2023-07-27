from query_graph import config

import numpy as np

import openai
openai.api_key = config.OPENAI_APIKEY()

import tiktoken

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

def embed_sentences(sentences):
   response = openai.Embedding.create(
      input=sentences,
      model="text-embedding-ada-002"
   )
   embeddings = list(np.array(response['data'][i]["embedding"]) for i in range(len(sentences)))
   return embeddings
    
if __name__ == "__main__":
   rspns = embed_sentences(["What day is it today?", "I hate weather forecasts"])
   print(rspns)