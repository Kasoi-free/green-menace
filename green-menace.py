from diffusers import DiffusionPipeline
import torch
from itertools import count

# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.enable_model_cpu_offload()
# comment out the line above and uncomment two lines below if your GPU has 20GB or more VRAM
#base.to("cuda")
#base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.enable_model_cpu_offload()
# comment out the line above and uncomment two lines below if your GPU has 20GB or more VRAM
#refiner.to("cuda")
#refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

prompt = "Shrek is love, Shrek is life"

# start the maddness over and over
for i in count(0):
	image = base(
		prompt=prompt,
		num_inference_steps=n_steps,
		denoising_end=high_noise_frac,
		output_type="latent",
	).images
	image = refiner(
		prompt=prompt,
		num_inference_steps=n_steps,
		denoising_start=high_noise_frac,
		image=image,
	).images[0]

	image.save(f"{i}_menace.png")
