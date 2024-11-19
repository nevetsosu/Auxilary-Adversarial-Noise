from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import os
import time
import sys
from model_wrappers import gemini, llama, gpt

# sleeps for a bit and prints while it does
def snooze(s):
    for _ in range(0, s):
        time.sleep(1)
        print("zzz")

class autoprompt:
     def __init__(self, model_name):
          self.init_model(model_name)
          self.init_db()
     # intialize the different model wrappers
     def init_model(self, model_name):
          if model_name == 'gemini':
               self.model = gemini()
               self.model_name = 'gemini'
          elif model_name == 'gpt':
               self.model = gpt()
               self.model_name = 'gpt'
          elif model_name == 'llama':
               self.model = llama()
               self.model_name = 'llama'
          else:
               raise ValueError('model_name can only be \'gemini\', \'gpt\', or \'llama\'')

     # save to database (csv)
     def save_data(self, new_data):
          if len(new_data) > 0:
               print("saving db")

               new_db = pd.DataFrame(new_data, columns=self.COLUMNS)
               final_db = pd.concat([self.db, new_db], ignore_index=True)
               final_db.to_csv(self.DB_PATH, index=False)
          else:
               print("no new prompts, db remains the same")

     # load the database file if it exists at DB_PATH
     # else create a new one at DB_PATH
     @staticmethod
     def load_db(path, columns):
          if os.path.exists(path):
               db = pd.read_csv(path, index_col=False)
          else:
               print(f"DB_PATH: {path} cannot be accessed, creating a new db")
               db = pd.DataFrame(columns=columns)

          return db

     # set up variables for data base usage, including loading the database itself
     def init_db(self):
          self.LOGO_DIR = Path(os.getenv("LOGO_DIR"))
          self.ALTERED_DIR = Path(os.getenv("LOGO_DIR"))
          self.COLUMNS = ['model', 'filename', 'name', 'response', 'classification']
          self.DB_PATH = os.getenv("DB_PATH")
          self.db = self.load_db(self.DB_PATH, ['index'] + self.COLUMNS)

     # model should be one of the classes in the model_wrappers module 
     # path should be a Path() object
     # timeout is the number of seconds it will wait if it errors out on call to the model (likely a rate_limit error)
     # If it gets an error, it will attempt to retry at max MAX_RETRIES times.
     @staticmethod
     def prompt(model, path, timeout=15, max_retries=10):
          retries = 0
          while (retries < max_retries):
               print(f'prompting for {path.name}')
               try:
                    response = model.prompt(path)
               except Exception as e:
                    print(e)
                    snooze(timeout)
                    continue

               # correct classification?
               logo_name = path.stem.split(".")[0]                                                # the logo name is extracted from the FILE NAME itself
               match = response.strip().lower().find(logo_name.lower()) != -1                        # checks if the response has the logo name in it 
               return (path.name, logo_name, response, match)                    # return results to be put in the database 


     # starts the auto prompt process to prompt files not-yet prompted
     def start(self):
          new_data = []
          # goes through each file in the LOGO_DIR with the suffix png or jpg
          for dir in [self.LOGO_DIR, self.ALTERED_DIR]:
               for entry in Path(dir).iterdir():
                    self.db = self.load_db(self.DB_PATH, self.COLUMNS)
                    # skips if non jpg or png file
                    if entry.suffix == ".jpg" and entry.suffix == ".png":
                         continue

                    # skips if entry is already in the database
                    filtered = self.db[(self.db['model'] == self.model_name) & (self.db['filename'] == entry.name)]
                    if not filtered.empty:
                         print(f'{entry.name} already present. skipping.')
                         continue

                    db_entry = self.prompt(self.model, self.LOGO_DIR / Path(entry.name))
                    new_data.append((self.model_name,) + db_entry)
                    self.save_data(new_data)
                    new_data = []

def usage():
     print("Usage: autoprompt MODEL")

def main():
     if (len(sys.argv) < 2):
          usage()
          exit()

     load_dotenv()
     model_name = sys.argv[1]
     if (model_name == "all"):
          for name in ["gemini", "gpt", "llama"]:
               auto_prompt = autoprompt(name) 
               auto_prompt.start()
     else:
          auto_prompt = autoprompt(model_name)
          auto_prompt.start()

if __name__ == "__main__":
    main()
