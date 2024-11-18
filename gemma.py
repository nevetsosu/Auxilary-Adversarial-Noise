from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch
import torch.nn as nn

"""
One time setup
"""
model_id = "google/paligemma-3b-mix-224"
device = "cuda:0"
dtype = torch.bfloat16

# convert logo to PIL
# logo_name = "Facebook"
# path = f"./logos/{logo_name}.png"
path = "./adversarial.png"
image = Image.open(path).convert('RGB')
image.save("./logo.png")

# setup model and processor
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    revision="bfloat16",
)
processor = PaliGemmaProcessor.from_pretrained(model_id)

# get target output
# target_text = logo_name
target_text = "cat"
target_inputs = processor.tokenizer(text=target_text + "<eos>", return_tensors="pt")['input_ids'][0].to(model.device)
target_length = target_inputs.size()[0]
decoded_label = processor.decode(target_inputs, skip_special_tokens=False)
print(f"decoded target label: {decoded_label}")  # DEBUG

"""
Sample Input
"""

# preprocess image
prompt = "<image>What is this"
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device, dtype)
model_inputs['pixel_values'].requires_grad = True

# using the non-wrapped generate avoids the @no_grad decorator on the normal generate
generation = model.generate.__wrapped__(
    model,
    **model_inputs,
    max_new_tokens=5,
    do_sample=False,
    output_scores=True,
    output_logits=True,
    return_dict_in_generate=True
)

# final sequence should only include newly generated tokens
input_len = model_inputs["input_ids"].shape[-1]
sequence = generation.sequences[0][input_len:]

# DEBUG
# decode output and checck if its correct
decoded = processor.decode(sequence, skip_special_tokens=False)
print(f'decoded prediction: {decoded}')

"""
Calculate Loss and Gradient to generate perturbation
"""

def iterative_FGSM(epsilon=0.01, iterations=1):
    # pixel_values = model_inputs['pixel_values'].clone().detach().to(device)
    # pixel_values.requires_grad = True

    # define the loss function
    loss_fn = nn.CrossEntropyLoss()

    for i in range(iterations):
        print(f"[iterative_FGSM] {i + 1} / {iterations}")
        # generate prediction
        generation = model.generate.__wrapped__(
            model,
            **model_inputs,
            # pixel_values=pixel_values,
            max_new_tokens=5,
            do_sample=False,
            output_scores=True,
            output_logits=True,
            return_dict_in_generate=True
        )

        # reshape logits and labels for the loss calculation
        logits = torch.stack(generation.logits[:target_length], dim=1)
        logits_reshaped = logits.view(-1, logits.size(-1))
        target_labels_reshaped = target_inputs.view(-1)

        # calculate loss and gradients
        loss = loss_fn(logits_reshaped, target_labels_reshaped)
        model.zero_grad()
        loss.backward()

        # create the adversarial pixel_values
        grad = model_inputs['pixel_values'].grad.data
        perturbed_pixel_values = model_inputs['pixel_values'].clone().detach().to(device) - epsilon * grad.sign()
        perturbed_pixel_values.requires_grad = True

        model_inputs['pixel_values'] = perturbed_pixel_values

    return model_inputs['pixel_values']

from torchvision import transforms
def tensor_to_image(processor, original_image, pixel_values):
    # assumed image mean and image std
    image_mean = [
        0.5,
        0.5,
        0.5
    ]
    image_std = [
        0.5,
        0.5,
        0.5
    ]

    # prepare mean, std, and rescale factor as tensors
    mean = torch.tensor(image_mean).view(-1,1,1).to(device)
    std = torch.tensor(image_std).view(-1,1,1).to(device)
    rescale = torch.tensor([255, 255, 255]).view(-1, 1, 1).to(device)

    # In order: denormalize, rescale by 255, convert to uint8
    denormed_tensor = pixel_values.squeeze(0) * std + mean
    rescaled_tensor = denormed_tensor * rescale
    int_tensor = rescaled_tensor.to(dtype=torch.uint8)
    img = transforms.ToPILImage()(int_tensor)

    # resize to original aspect ratio (lossy)
    # width, height = original_image.size
    # resized_img = img.resize((width, height))
    return img
    # return resized_img

adv_pixel_values = iterative_FGSM(iterations=10)
adv_img = tensor_to_image(processor.image_processor, image, adv_pixel_values)
adv_img.save("./adversarial.png")
