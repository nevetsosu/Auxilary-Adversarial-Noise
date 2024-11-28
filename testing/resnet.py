from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch
from datasets import load_dataset
from PIL import Image
dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]
image = Image.open('./logos/AT&T.png').convert("RGB")

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-152")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-152")

inputs = feature_extractor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])

generate = model.generate(**inputs)
print(generate)

