import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import os

def prompt_logo(model, filepath):
    file = genai.upload_file(filepath)
    result = model.generate_content(
        [file, "\n\n", "Give the name of this logo, just the name and nothing else."]
    )
    return result

def find_match(text, keyword): 
    return False if text.lower().find(keyword.lower()) == -1 else True

def load_db(path, columns):
    if os.path.exists(path):
        db = pd.read_csv(path)
    else:
        print(f"DB_PATH: {path} cannot be accessed, creating a new db")
        db = pd.DataFrame(columns=columns)
    db.set_index('filename', inplace=True)

    return db

def main():
    load_dotenv()
    LOGO_DIR = Path(os.getenv("LOGO_DIR"))

    # configure model
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")

    # load db
    COLUMNS = ['name', 'filename', 'response', 'classification']
    DB_PATH = os.getenv("DB_PATH")
    db = load_db(DB_PATH, COLUMNS)

    # prompt file if not already in db
    new_data = []
    for entry in Path(LOGO_DIR).iterdir():
        
        if (entry.name not in db.index):
            print(f'prompting for {entry.name}')
            try:
                response = prompt_logo(model, LOGO_DIR / Path(entry.name))
            except Exception as e:
                print(e)
                break
            print(f'response: |{response.text}|')

            # identify whether the logo was identified
            match = find_match(response.text, entry.stem)
            new_data.append((entry.stem, entry.name, response.text, match))
        else:
            print(f'{entry.name} already present. skipping.')

    # save new_data into db
    if len(new_data) > 0:
        print("saving db")
        
        new_db = pd.DataFrame(new_data, columns=COLUMNS)
        final_db = pd.concat([db, new_db], ignore_index=True)
        final_db.to_csv(DB_PATH)
    else:
        print("no new prompts, db remains the same")

if __name__ == "__main__":
    main()