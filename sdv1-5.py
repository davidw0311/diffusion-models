from diffusers import DiffusionPipeline
import torch
import os

pipeline = DiffusionPipeline.from_pretrained("./sd-v1-5", cache_dir=".", use_safetensors=True)
pipeline.to("mps")
pipeline.enable_attention_slicing()

prompts = [ "An image of a squirrel in Picasso style",
            "Macro photography of dewdrops on a spiderweb",
            "Underwater photography of a coral reef, with diverse marine life and a scuba diver for scale"
            ]
os.makedirs('images_sd1-5' , exist_ok=True)
for p in prompts:
    print(f'generating image for "{p}"')
    _ = pipeline(p, num_inference_steps=1)
    image = pipeline(p).images[0]
    name = p.replace(" ", "-")
    image.save(f"images_sd1-5/{name}.png")