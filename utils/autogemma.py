from pathlib import Path
import os
from dotenv import load_dotenv
from gemma import gemma

load_dotenv()

LOGO_DIR = Path(os.getenv("LOGO_DIR"))
ALTERED_DIR = Path(os.getenv("ALTERED_DIR"))

g = gemma()
print("finished initializing gemma ")
for entry in LOGO_DIR.iterdir():
    if (entry.suffix != ".png"):
        print("WARNING: found a non-.png file in the LOGO_DIR, ignoring")
        continue

    out_file = ALTERED_DIR / f"{entry.stem}-Adversarial.png"
    if (out_file.exists()):
        print(f"ignoring {entry.name} since {out_file.name} exists")
        continue
    print(f"perturbing {entry.name} to {out_file.name}")
    g.perturb(entry, out_file, "cat", 0.01, 10000, 0.5, True)
