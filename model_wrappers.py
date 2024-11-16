# gemini
import google.generativeai as genai
import os

prompt = "Give the name of this logo, just the name and nothing else."
class gemini:
     def __init__(self):
          genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
          self = genai.GenerativeModel("gemini-1.5-flash")

     def prompt(self, filepath):
          file = genai.upload_file(filepath)
          result = self.model.generate_content(
               [file, "\n\n", prompt]
          )
          file.delete()
          return result

# gpt
from openai import OpenAI
import base64

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class gpt:
     def __init__(self):
          self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

     def prompt(self, filePath):
          base64_image = encode_image(filePath)

          completion = self.client.chat.completions.create(
               model="gpt-4o-mini",
               messages=[
                    {
                         "role": "user",
                         "content": [
                              {
                                   "type": "text",
                                   "text": prompt,
                              },
                              {
                                   "type": "image_url",
                                   "image_url": {
                                   "url":  f"data:image/png;base64,{base64_image}"
                                   },
                              },
                         ],
                    }
               ])
          return completion.choices[0]

class llama:
     def __init__(self):
          self.client = OpenAI(
          api_key =os.getenv("LLAMA_API_KEY"),
          base_url = "https://api.llama-api.com"
          )

     def prompt(self, filePath):
          base64_image = encode_image(filePath)

          completion = self.client.chat.completions.create(
               model="llama3.2-90b-vision",
               messages=[
                    {
                         "role": "user",
                         "content": [
                              {
                                   "type": "text",
                                   "text": prompt,
                              },
                              {
                                   "type": "image_url",
                                   "image_url": {
                                   "url":  f"data:image/png;base64,{base64_image}"
                                   },
                              },
                         ],
                    }
               ])
          return completion.choices[0].message.content