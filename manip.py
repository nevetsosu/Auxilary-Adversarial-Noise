import numpy as np
from PIL import Image, ImageEnhance
from dotenv import load_dotenv
from pathlib import Path
import os

def saturate_image(input_path, output_path, saturation_factor=1.0):
    """
    Adjusts the saturation of an image and saves the modified image.
    
    Parameters:
        input_path (str): The file path of the input image.
        output_path (str): The file path where the modified image will be saved.
        saturation_factor (float): The factor by which to enhance saturation.
                                    1.0 means no change, <1.0 decreases saturation,
                                    and >1.0 increases saturation.
    """
    # Step 1: Open the image file
    img = Image.open(input_path)
    
    # Step 2: Create an enhancer for color (saturation)
    enhancer = ImageEnhance.Color(img)
    
    # Step 3: Enhance saturation
    img_saturated = enhancer.enhance(saturation_factor)
    
    # Step 4: Save the modified image
    img_saturated.save(output_path)
    print(f"Saturated image saved as: {output_path}")

def add_noise_to_logo(logo_path, output_path, noise_level=0.2):
    # Load the logo image
    logo = Image.open(logo_path).convert("RGB")
    
    # Convert the logo to a numpy array
    logo_array = np.array(logo)
    
    # Generate random noise
    noise = np.random.randint(0, 256, logo_array.shape, dtype=np.uint8)
    
    # Combine logo and noise
    # Blend the logo with the noise
    noisy_logo_array = logo_array.astype(float) + noise_level * noise.astype(float)
    
    # Clip values to ensure they are valid pixel values (0-255)
    noisy_logo_array = np.clip(noisy_logo_array, 0, 255).astype(np.uint8)
    
    # Convert back to a PIL Image
    noisy_logo = Image.fromarray(noisy_logo_array, "RGB")
    
    # Save the resulting image
    noisy_logo.save(output_path)

def main():
    load_dotenv()
    LOGO_DIR = Path(os.getenv("LOGO_DIR"))

    # Noise
    for entry in Path(LOGO_DIR).iterdir():
        if entry.suffix != ".png" and entry.suffix != ".jpg":
            continue
        if "Noise" in entry.stem:
            continue

        for noise_level in range(1, 10, 1):
            new_name = entry.stem + f".Noise{noise_level}" + entry.suffix
            if os.path.exists(LOGO_DIR / Path(new_name)):
                continue
            print(f"working on {new_name}")
            add_noise_to_logo(LOGO_DIR / Path(entry.name), LOGO_DIR / Path(new_name), noise_level * 0.1)
        
# Usage
# add_noise_to_logo("./logos/Google.jpg", "./logos/Google.Noisy6.jpg", noise_level=0.6)
# saturate_image("./logos/Google.jpg", "./logos/Google.Saturate.jpg", 1.5)  # Adjust the saturation as needed

if __name__ == "__main__":
    main()