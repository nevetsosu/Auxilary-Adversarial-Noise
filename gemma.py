from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch

model_id = "google/paligemma-3b-mix-224"
device = "cuda:0"
dtype = torch.bfloat16

image = Image.open('./logos/Facebook.png').convert('RGB')
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    revision="bfloat16",
).eval()
processor = AutoProcessor.from_pretrained(model_id)

# Instruct the model to create a caption in Spanish
prompt = " <image>Give me the logo name and only the name."
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device, dtype)
print(f"model_inputs: {model_inputs}")
model_inputs['pixel_values'].requires_grad = True

input_len = model_inputs["input_ids"].shape[-1]

# using the non-wrapped generate forces off any grad calculation tracking
generation = model.generate.__wrapped__(
    model,
    **model_inputs,
    max_new_tokens=5,
    do_sample=False,
    output_scores=True,
    output_logits=True,
    return_dict_in_generate=True
)

print(generation.logits)
sequence = generation.sequences[0][input_len:]
# print(f"generation {generation}")
decoded = processor.decode(sequence, skip_special_tokens=False)
print(f'decoded prediction: {decoded}')
#
# ADVERSARIAL TESTING
#

# get target inputs
target_label = "facebook"
target_inputs = processor.tokenizer(text=target_label + "<eos>", return_tensors="pt")['input_ids'][0].to(model.device)
target_length = target_inputs.size()[0]

# debug print
decoded_label = processor.decode(target_inputs, skip_special_tokens=False)
print(decoded_label)

# loss calculation
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()

print(f"GENERATION LOGITS: {generation.logits}")
logits = torch.stack(generation.logits[:target_length], dim=1)
print(f"logits after stack: {logits}")
print(f"size before: {len(generation.logits)}")
print(f"size after: {logits.size()}")
print(target_inputs)

logits_reshaped = logits.view(-1, logits.size(-1))
target_labels_reshaped = target_inputs.view(-1)

print(f"l reshaped: {logits_reshaped}")
print(f"target_labels_reshaped: {target_labels_reshaped}")

loss = loss_fn(logits_reshaped, target_labels_reshaped)

model.zero_grad()
loss.backward()

# adjust original data
epsilon = 0.1
grad = model_inputs['pixel_values'].grad.data
adversarial = model_inputs['pixel_values'] + epsilon * grad.sign()

print(loss)
# target_caption = "<image>a cat"
# target_inputs = processor(text=target_caption, images=image, return_tersors="pt").to(model device, )
