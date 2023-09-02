from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", cache_dir=".")
pipeline.save_pretrained('sd-v1-5')

print('model downloaded!')
