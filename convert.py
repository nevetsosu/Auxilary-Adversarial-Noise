from PIL import Image
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: convert file1 [file2] [file3]...")

paths = [Path(s) for s in sys.argv[1:]]

for path in paths:
    img = Image.open(path)
    img.save(path.with_suffix(".png"), "PNG")

