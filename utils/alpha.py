from PIL import Image
import sys
from pathlib import Path
import numpy as np

if len(sys.argv) < 2:
    print("Usage: convert file1 [file2] [file3]...")

paths = [Path(s) for s in sys.argv[1:]]

for path in paths:
     if path.suffix != ".png": continue

     image = Image.open(path)
     image = image.convert("RGBA")

     image_array = np.array(image)

     # Extract the RGBA channels
     r, g, b, a = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2], image_array[:, :, 3]

     # Define the background color (e.g., white background)
     background_color = [255, 255, 255]  # White

     # Normalize the alpha channel to the range [0, 1]
     alpha_normalized = a / 255.0

     # Blend the image with the background color using the alpha channel
     r_blended = (alpha_normalized * r + (1 - alpha_normalized) * background_color[0]).astype(np.uint8)
     g_blended = (alpha_normalized * g + (1 - alpha_normalized) * background_color[1]).astype(np.uint8)
     b_blended = (alpha_normalized * b + (1 - alpha_normalized) * background_color[2]).astype(np.uint8)

     # Stack the RGB channels to create the final image
     final_image_array = np.stack([r_blended, g_blended, b_blended], axis=-1)

     # Convert the array back to an image
     final_image = Image.fromarray(final_image_array)
     final_image.save(path)
