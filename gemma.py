from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch

model_id = "google/paligemma-3b-mix-224"
device = "cuda:0"
dtype = torch.bfloat16

image = Image.open('./logos/Apple.png').convert('RGB')
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    revision="bfloat16",
).eval()
processor = AutoProcessor.from_pretrained(model_id)

# Instruct the model to create a caption in Spanish
prompt = " <image>caption this"
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device, dtype)
input_len = model_inputs["input_ids"].shape[-1]

generation = model.generate.__wrapped__(model, **model_inputs, max_new_tokens=20, do_sample=False)
generation = generation[0][input_len:]
decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)

