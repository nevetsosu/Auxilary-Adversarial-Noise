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

# Generating altered
LOGO_DIR and ALTERED_DIR should be defined in a .env.
Your images should already be a PNG and should be in LOGO_DIR.
Your images should also already be checked by the resolution tool.

1. Run utils/manip.py

The script will look in LOGO_DIR and the results will be in ALTERED_DIR.
Files that have already had generated variants will be skipped.

# Classifying
After all the preprocessing: conversion and resolution assurance.
utils/prompt.py allows for prompting a single specific model.

utils/autoprompt.py allows for prompting all files on a specific model or all models. It will attempt to prompt all unprompted files in LOGO_DIR and ALTERED_DIR. autoprompt will use the csv at DB_PATH to determine what has already been prompted and save new prompts to there. If the csv at DB_PATH doesn't exist already, it will be created. DB_PATH should be specified in the .env. 

# Todo
In gemma, I should output either the accumulated perturbation or the average perturbation.
This allows me to create different images that may have different amounts of perturbation (using a multplier), but still from the same perturbation pattern. 

Handle Transparency by replacing it with white RGB pixels

Improve imperceptibility of adversarial examples. 

Find the threshold between human perceptility and model perceptibility.
Make charts

# Charts
Find the perceptiability thresholds for humans and models. 

Present more details about how the perturbations were added.
We have 5 levels of both saturation and noise.

Adversarial noise generation should include the learning rate (epsilon) and the loss threshold.

Should we even bother to present accuracy numbers??

We should add peligemma to the charts, which will be the only ones that can actually misclassify. 


