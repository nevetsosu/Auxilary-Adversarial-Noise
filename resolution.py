from PIL import Image
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: convert file1 [file2] [file3]...")

paths = [Path(s) for s in sys.argv[1:]]

for path in paths:
    img = Image.open(path)
    w, h = img.size

    scale = 1
    max_side = max(w, h)
    min_side = min(w, h)
    if max_side >= 2000:
        scale = 2000 / max_side

    if scale * min_side >= 768:
        scale = scale * 768 / min_side
    else:
        continue

    w_new = int(w * scale)
    h_new = int(h * scale)
    print(f"Scaling {path} from ({w}, {h}) to ({w_new}, {h_new})")
    
    new_img = img.resize((w_new, h_new), Image.LANCZOS)
    new_img.save(path)

