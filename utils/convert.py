from PIL import Image
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
LOGO_DIR = Path(os.getenv("LOGO_DIR"))
ALTERED_DIR = Path(os.getenv("ALTERED_DIR"))

for entry in LOGO_DIR.iterdir():
     if entry.suffix == ".png":
          continue

     img = Image.open(entry)
     img.save(ALTERED_DIR / entry.name, "PNG")

