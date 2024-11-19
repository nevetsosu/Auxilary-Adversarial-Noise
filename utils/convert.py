from PIL import Image
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
LOGO_DIR = Path(os.getenv("LOGO_DIR"))
ALTERED_DIR = Path(os.getenv("ALTERED_DIR"))

def convert(dir:Path):

    for entry in dir.iterdir():
        if entry.suffix == ".png":
            continue
        elif entry.suffix == ".jpg":
            if delete:
                entry.unlink()
            continue

        img = Image.open(entry)
        img.save(ALTERED_DIR / entry.name, "PNG")

