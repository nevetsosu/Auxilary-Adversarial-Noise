# Load model directly
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "microsoft/Florence-2-base"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
).to(device)
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
)

task = "<CAPTION>"

image = Image.open('./logos/DHL.png').convert('RGB')

inputs = processor(text=task, images=image, return_tensors='pt').to(device, torch_dtype)

generated_ids = model.generate.__wrapped__(model, **inputs)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
parsed_answer = processor.post_process_generation(generated_text, task=task, image_size=(image.width, image.height))

print(parsed_answer)
