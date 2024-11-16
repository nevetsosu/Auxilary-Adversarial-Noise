from PIL import Image

def blend(image1_path, image2_path, output_path, alpha=0.3):
     """
     Combine two images with a weighted factor using Pillow.

     Args:
          image1_path (str): Path to the first image (base image).
          image2_path (str): Path to the second image (overlay image).
          output_path (str): Path to save the combined image.
          alpha (float): Weight for the overlay image. Should be in the range [0, 1].
                         The base image gets (1 - alpha).
     """
     # Open the images
     image1 = Image.open(image1_path).convert("RGBA")
     image2 = Image.open(image2_path).convert("RGBA")

     # Resize the second image to match the first image's dimensions (if needed)
     if image1.size != image2.size:
          image2 = image2.resize(image1.size, Image.LANCZOS)

     # Blend the images
     blended = Image.blend(image1, image2, alpha)

     # Save the result
     blended.save(output_path)
     print(f"Combined image saved to {output_path}")

# Example usage
blend(
image1_path="./logos/FedEx.png",
image2_path="./elijah.png",
output_path="./p.png",
alpha=0.5  # Adjust the weight of the overlay image
)
