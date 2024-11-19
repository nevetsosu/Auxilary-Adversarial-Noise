from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch
import torch.nn as nn
import sys
from torchvision import transforms

class gemma:
    def __init__(self):
        model_id = "google/paligemma-3b-mix-224"
        self.device = "cuda:0"
        self.dtype = torch.bfloat16

        # setup model and processor 
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map=self.device,
            revision="bfloat16",
        )
        self.processor = PaliGemmaProcessor.from_pretrained(model_id)

    def tensor_to_image(self, original_image, pixel_values):
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
        mean = torch.tensor(image_mean).view(-1,1,1).to(self.device)
        std = torch.tensor(image_std).view(-1,1,1).to(self.device)
        rescale = torch.tensor([255, 255, 255]).view(-1, 1, 1).to(self.device)

        # In order: denormalize, rescale by 255, convert to uint8
        denormed_tensor = pixel_values.squeeze(0) * std + mean
        rescaled_tensor = denormed_tensor * rescale
        int_tensor = rescaled_tensor.to(dtype=torch.uint8)

        # convert to PIL image
        img = transforms.ToPILImage()(int_tensor)

        # resize to original resolution
        # width, height = original_image.size
        # img = img.resize((width, height))

        # restore alpha is the original had an alpha
        # if (original_image.mode == "RGBA"):
        #    r, g, b = img.split()
        #    _,_,_, a = original_image.split()
        #    img = Image.merge("RGBA", (r, g, b, a))

        return img

    @staticmethod
    def total_variation_loss(x):
        # Calculate the total variation loss for an image tensor x
        batch_size, channels, height, width = x.size()

        # Calculate the difference between neighboring pixels in height and width directions
        diff_x = torch.diff(x, dim=3)  # Difference along width (columns)
        diff_y = torch.diff(x, dim=2)  # Difference along height (rows)

        # Compute L2 norm of the differences (this encourages smoothness)
        tv_loss = torch.sum(diff_x**2) + torch.sum(diff_y**2)

        # Normalize by the number of pixels (height * width) in the image
        tv_loss = tv_loss / (batch_size * channels * height * width)

        return tv_loss

    def iterative_FGSM(self, image_rgb, target_label, epsilon=0.01, iterations=1, loss_threshold=0.0, progressive_save=False, out_path=None):
        if progressive_save and out_path == None:
            raise ValueError("out_path should be given if progressive_save is true")

        target_inputs = self.processor.tokenizer(text=target_label + "<eos>", return_tensors="pt")['input_ids'][0].to(self.device)
        target_length = target_inputs.size()[0]
        decoded_label = self.processor.decode(target_inputs, skip_special_tokens=False)
        print(f"decoded target label: {decoded_label}")  # DEBUG

        """
        Sample Input
        """

        # preprocess image
        prompt = "<image>What is this"
        model_inputs = self.processor(text=prompt, images=image_rgb, return_tensors="pt").to(self.device, self.dtype)
        model_inputs['pixel_values'].requires_grad = True

        # using the non-wrapped generate avoids the @no_grad decorator on the normal generate
        generation = self.model.generate.__wrapped__(
            self.model,
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
        decoded = self.processor.decode(sequence, skip_special_tokens=False)
        print(f'decoded prediction: {decoded}')

        # define the loss function
        loss_fn = nn.CrossEntropyLoss()
        reg_loss_fn = self.total_variation_loss
        for i in range(iterations):
            print(f"[iterative_FGSM] {i + 1} / {iterations}")
            # generate prediction
            generation = self.model.generate.__wrapped__(
                self.model,
                **model_inputs,
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
            # loss = loss_fn(logits_reshaped, target_labels_reshaped)
            loss = loss_fn(logits_reshaped, target_labels_reshaped) + reg_loss_fn(model_inputs['pixel_values'])
            self.model.zero_grad()
            loss.backward()
            print(f"loss: {loss}")
            # create the adversarial pixel_values
            grad = model_inputs['pixel_values'].grad.data
            perturbed_pixel_values = model_inputs['pixel_values'].clone().detach().to(self.device) - epsilon * grad.sign()
            perturbed_pixel_values = torch.clamp(perturbed_pixel_values, -1, 1)
            perturbed_pixel_values.requires_grad = True

            model_inputs['pixel_values'] = perturbed_pixel_values

            if (progressive_save):
                self.tensor_to_image(image_rgb, model_inputs['pixel_values']).save(out_path)

            if (loss < loss_threshold):
                print("loss below loss threshold, early break")
                break

        return model_inputs['pixel_values']

    def perturb(self, in_path, out_path, target_label, epsilon, iterations, loss_threshold, progressive_save):
        """
        One time setup
        """
        # open image
        image = Image.open(in_path)
        image_rgb = image.convert("RGB")

               # get target output
        adv_pixel_values = self.iterative_FGSM(
            image_rgb,
            target_label,
            epsilon=epsilon,
            iterations=iterations,
            loss_threshold=loss_threshold,
            progressive_save=progressive_save,
            out_path=out_path
        )
        adv_img = self.tensor_to_image(image, adv_pixel_values)
        adv_img.save(out_path)


def main():
    if len(sys.argv) != 8:
        print("Usage: gemma IN_PATH OUT_PATH TARGET_LABEL EPSILON ITERATIONS LOSS_THRESHOLD PROGRESSIVE_SAVE")
        exit() 
    _, in_path, out_path, target_label, epsilon, iterations, loss_threshold, progressive_save = sys.argv

    epsilon = float(epsilon)
    iterations = int(iterations)
    loss_threshold = float(loss_threshold)
    progressive_save = progressive_save.lower() == "true"

    print(f'in_path: {in_path}')
    print(f'out_path: {out_path}')
    print(f'target_label: {target_label}')
    print(f'epsilon: {epsilon}')
    print(f'iterations: {iterations}')
    print(f'loss_threshold: {loss_threshold}')
    print(f'progressive_save: {progressive_save}')

    gemma().perturb(in_path, out_path, target_label, epsilon, iterations, loss_threshold, progressive_save) 

if __name__ == "__main__":
    main()
