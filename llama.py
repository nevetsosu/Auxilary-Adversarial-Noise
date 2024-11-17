from PIL import Image
from transformers import MllamaProcessor, MllamaForConditionalGeneration
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'using device: {device}')

model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

print("initializing processor")
processor = MllamaProcessor.from_pretrained(model_name)

print("initializing model")
model = MllamaForConditionalGeneration.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision",
    device_map="auto",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
prompt = "Give the name of this logo, just the name and nothing else."
messages = [
    [
        {
            "role": "user", 
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ],
]

text = processor.apply_chat_template(messages, add_generation_prompt=True)

image = Image.open('./logos/AT&T.png').convert('RGB')

print("processing inputs")
inputs = processor(text=text,images=image, return_tensors='pt').to(model.device)

print("forwarding")
outputs = model.generate(**inputs, max_new_tokens=50)
print(f'output.scores[0].requires_grad: {output.scores[0].requires_grad}')

final = processor.batch_decode(outputs)
print("decoding")
#generated = processor.decode(outputs[0])

# print("generated")
# print(generated)


