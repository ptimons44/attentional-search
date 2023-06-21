import openai
import config
openai.api_key = config.OPENAI_APIKEY()

def callGPT(prompt, max_tokens=0):
    response = openai.ChatCompletion.create(
      model=config.OPENAI_MODEL(),
      messages=[
         {"role": "user", "content": prompt}
      ]
   )
    return response.choices[0].message.content