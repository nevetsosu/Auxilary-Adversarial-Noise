from dotenv import load_dotenv
import pandas as pd
import os

def process_model(db):
    total_true = db['classification'].sum()
    total_count = db['classification'].count()
    print(total_true)
    print(total_count)
load_dotenv()
db = pd.read_csv(os.getenv("DB_PATH"))
print(db.columns)

gemini_db = db[db['model'] == 'gemini']
gpt_db = db[db['model'] == 'gpt']
llama_db = db[db['model'] == 'llama']

process_model(gemini_db)
process_model(gpt_db)
process_model(llama_db)
