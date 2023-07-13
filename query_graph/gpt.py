from query_graph import config

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
    
if __name__ == "__main__":
   rspns = callGPT("What day is it today?")
   print(rspns)