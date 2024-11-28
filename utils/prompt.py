from model_wrappers import gpt, llama, gemini
from autoprompt import autoprompt
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def usage():
    print("Usage: prompt MODEL FILE")

if (len(sys.argv) < 3):
    usage()

model_name = sys.argv[1]
path = Path(sys.argv[2])

if model_name == 'gemini':
    model = gemini()
elif model_name == 'llama':
    model = llama()
elif model_name == 'gpt':
    model = gpt()
else:
    print("MODEL should be 'gpt', 'llama', or 'gemini'")
    exit()

if not path.exists():
    print("Invalid path")
    exit()

response = autoprompt.prompt(model, path)
print(response)
