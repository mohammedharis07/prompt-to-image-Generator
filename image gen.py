import torch
from diffusers import StableDiffusionPipeline

# ✅ Available models for different styles
MODEL_OPTIONS = {
    "realistic": "runwayml/stable-diffusion-v1-5",
    "anime": "hakurei/waifu-diffusion",
    "fantasy": "stabilityai/stable-diffusion-2-1",
    "pixel_art": "nitrosocke/stable-diffusion-pixel-art"
}

# ✅ User selects the model style
print("Choose an image style: realistic, anime, fantasy, pixel_art")
style = input("Enter style: ").strip().lower()

# Default to realistic if an invalid option is given
model_name = MODEL_OPTIONS.get(style, MODEL_OPTIONS["realistic"])

# ✅ Load the selected model
print(f"Loading model: {model_name} ...")
pipe = StableDiffusionPipeline.from_pretrained(model_name)

# Automatically choose CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)

# ✅ Get user input for the image prompt
prompt = input("Enter your image description: ")
negative_prompt = input("Enter negative prompt (or press Enter to skip): ")

# ✅ Set generation parameters
guidance_scale = float(input("Enter guidance scale (Default 7.5, higher = more stylized): ") or 7.5)
num_steps = int(input("Enter number of inference steps (Default 50, higher = better quality): ") or 50)

# ✅ Generate the image
print("Generating image... This may take a few seconds.")
image = pipe(prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, num_inference_steps=num_steps).images[0]

# ✅ Save and display the image
image_path = "generated_image.png"
image.save(image_path)
image.show()

print(f"✅ Image saved as {image_path}")
