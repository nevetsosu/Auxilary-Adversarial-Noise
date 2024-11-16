import google.generativeai as genai
import os

class gemini:
     def __init__(self):
          genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
          self = genai.GenerativeModel("gemini-1.5-flash")

     def prompt(self, filepath):
          file = genai.upload_file(filepath)
          result = self.model.generate_content(
               [file, "\n\n", "Give the name of this logo, just the name and nothing else."]
          )
          file.delete()
          return result