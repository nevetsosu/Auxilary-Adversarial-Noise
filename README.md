# Classification Pipeline
1. Convert jpg to png. This can be skipped if PNG already.
2. Ensure resolution sizes are within bounds.
3. Classify original, if not recognized, remove image and do not continue.
4. Generate altered logos
5. Classify altered

Instead of classifying the original and altered logos in two different steps (3 and 5), you can skip step 3 and the original image will be classified in step 5 along with the altered logos.
Classifying the original first just shows if the original is even recognized, if the original isn't recognized, perturbations are not going to matter.

# Converting JPG to PNG
LOGO_DIR and ALTERED_DIR should be defined in a .env, as well as the model api keys.
You can specify whether the original image is deleted.

1. Put your image file (jpg or png) into the LOGO_DIR. 
2. Run utils/convert.py

# Ensuring resolution
LOGO_DIR and ALTERED_DIR should be defined in a .env, as well as the model api keys.
Your image should already be a PNG and should be in LOGO_DIR.
It WILL replace the original image.

1. Run utils/resolution.py

# Generate altered logos
manip.py will look into LOGO_DIR for unaltered logos. The file names should be in the form ``[LOGO_NAME].png``.

Manipulated logos will be generated and put into the same LOGO_DIR. The generated file names will be in the form ``[LOGO_NAME].[MANIPULATION_TYPE][MANIPULATION_STRENGTH].png``.

prompt.py expects both unaltered logo files and altered logo files to be in this format.


# Todo
Handle Transparency by replacing it with white RGB pixels
