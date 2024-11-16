from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import os
import time

# our files
from model_wrappers import gemini

# sleeps for a bit and prints while it does
def snooze(s):
    for _ in range(0, s):
        time.sleep(1)
        print("zzz")

class auto_prompt:
     def __init__(self):
          self.init_models()
          self.init_db()

     # intialize the different model wrappers
     def init_models(self):
          self.gemini = gemini()

     # save to database (csv)
     def save_data(self, new_data):
          if len(new_data) > 0:
               print("saving db")

               new_db = pd.DataFrame(new_data, columns=self.COLUMNS)
               new_db.set_index('filename', inplace=True)
               final_db = pd.concat([self.db, new_db])
               final_db.to_csv(self.DB_PATH)
          else:
               print("no new prompts, db remains the same")

     # load the database file if it exists at DB_PATH
     # else create a new one at DB_PATH
     @staticmethod
     def load_db(path, columns):
          if os.path.exists(path):
               db = pd.read_csv(path)
          else:
               print(f"DB_PATH: {path} cannot be accessed, creating a new db")
               db = pd.DataFrame(columns=columns)
          db.set_index('filename', inplace=True)

          return db

     # set up variables for data base usage, including loading the database itself
     def init_db(self):
          self.LOGO_DIR = Path(os.getenv("LOGO_DIR"))
          self.COLUMNS = ['name', 'filename', 'response', 'classification']
          self.DB_PATH = os.getenv("DB_PATH")
          self.db = self.load_db(self.DB_PATH, self.COLUMNS)

     # substring search for KEYWORD in TEXT
     @staticmethod
     def find_match(text, keyword):
          return False if text.lower().find(keyword.lower()) == -1 else True

     # starts the auto prompt process to prompt files not-yet prompted
     def start(self):
          new_data = []

          # goes through each file in the LOGO_DIR with the suffix png or jpg
          for entry in Path(self.LOGO_DIR).iterdir():

               # skips if non jpg or png file
               if entry.name.find("jpg") == -1 and entry.name.find("png") == -1:
                    continue

               # skips if entry is already in the database
               if (entry.name in self.db.index):
                    print(f'{entry.name} already present. skipping.')
                    continue

               # if we get a rate_limit error, it is caught, we sleep for 15, then we try again.
               # we keep doing this til the rate limit is taken off and the file is properly prompted
               while (1):
                    print(f'prompting for {entry.name}')
                    try:
                         response = gemini.prompt(self.LOGO_DIR / Path(entry.name))
                    except Exception as e:
                         # we assume that the exception here was because of the rate-limit
                         print(e)
                         snooze(15)
                         continue
                    print(f'response: |{response.text}|')

                    # identify whether the logo was identified
                    name = entry.stem.split(".")[0]                                  # the logo name is extracted from the FILE NAME itself
                    match = find_match(response.text.strip(), name)                  # checks if the response has the logo name in it 
                    new_data.append((name, entry.name, response.text, match))        # adds result to database

                    break

def main():
     load_dotenv()
     auto_prompter = auto_prompt()
     auto_prompter.start()

if __name__ == "__main__":
    main()